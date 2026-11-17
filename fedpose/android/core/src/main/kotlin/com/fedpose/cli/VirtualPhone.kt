package com.fedpose.cli

import com.fedpose.core.*
import java.io.File
import kotlin.system.exitProcess

/**
 * JVM command line harness around the on-device code, used for parity tests and for developing
 * without a phone.  Commands:
 *
 *   selftest
 *   features --session f.csv --features acc_grav            -> feature rows on stdout
 *   dataset  --root R --client A --split train --features acc_grav --out out.csv
 *   clip     --session f.csv --features acc_grav --weights w.txt [--starts 0.5,1.0] [--offset-s 3.0]
 *            -> blind-test style prediction of a 2 s clip: window count, mean probabilities, label
 *   train    --data out.csv --features acc_grav --epochs 5 --lr 0.3 --batch 32 --seed 1 --out w.txt
 *   phone    --server http://127.0.0.1:8080 --client A --root R --features acc_grav
 *            [--epochs 2 --lr 0.3 --batch 32 --seed 0 --codec f32|int8 --adaptive --threshold 1.0
 *             --mu 0.0 --out outdir
 *             --drop-rounds 3-4 --drop-ms 5000]   (fake a dropout: delay the upload in those rounds)
 */
fun main(argv: Array<String>) {
    if (argv.isEmpty()) { System.err.println("usage: see VirtualPhone.kt header"); exitProcess(2) }
    val cmd = argv[0]
    val a = Args(argv.drop(1))
    when (cmd) {
        "selftest" -> selfTest()
        "features" -> {
            val s = Features.loadSessionCsv(File(a.str("session")))
            Features.windowFeatures(s, a.str("features", Constants.DEFAULT_FEATURE_SET))
                .forEach { println(it.joinToString(",")) }
        }
        "dataset" -> {
            val d = Dataset.load(File(a.str("root")), a.str("client"), a.str("split"), a.str("features")) { System.err.println(it) }
            File(a.str("out")).printWriter().use { w ->
                for (i in 0 until d.size) w.println(d.y[i].toString() + "," + d.X[i].joinToString(","))
            }
            println("wrote ${d.size} windows")
        }
        "train" -> {
            val d = readFeatureCsv(File(a.str("data")))
            val nf = Constants.nFeatures(a.str("features"))
            val m = SoftmaxModel(nf)
            val cfg = TrainConfig(a.int("epochs", 2), a.dbl("lr", 0.3), a.int("batch", 32), a.dbl("l2", 0.0), a.dbl("mu", 0.0))
            val w = m.train(SoftmaxModel.initWeights(nf), d, cfg, a.long("seed", 0))
            File(a.str("out")).writeText(w.joinToString("\n") { it.toString() } + "\n")
        }
        "clip" -> {
            val s = Features.loadSessionCsv(File(a.str("session")))
            val fs = a.str("features", Constants.DEFAULT_FEATURE_SET)
            val w = File(a.str("weights")).readLines().filter { it.isNotBlank() }.map { it.toDouble() }.toDoubleArray()
            val starts = a.str("starts", "0.5,1.0").split(',').map { it.toDouble() }.toDoubleArray()
            val t0 = BlindTest.clipStart(s, fs) ?: error("session lacks a sensor needed by $fs")
            val wins = BlindTest.clipWindows(s, fs, t0 + Math.round(a.dbl("offset-s", 0.0) * 1e9), starts)
            val p = BlindTest.clipProba(SoftmaxModel(Constants.nFeatures(fs)), w, wins)
            println("windows=${wins.size}")
            println(if (p == null) "none" else p.joinToString(",") + "\nlabel=" + BlindTest.argmax(p))
        }
        "phone" -> phone(a)
        else -> { System.err.println("unknown command $cmd"); exitProcess(2) }
    }
}

private class Args(list: List<String>) {
    private val m = HashMap<String, String>()
    init {
        var i = 0
        while (i < list.size) {
            val k = list[i].removePrefix("--")
            if (i + 1 < list.size && !list[i + 1].startsWith("--")) { m[k] = list[i + 1]; i += 2 } else { m[k] = "true"; i++ }
        }
    }
    fun str(k: String, d: String? = null) = m[k] ?: d ?: error("missing --$k")
    fun int(k: String, d: Int) = m[k]?.toInt() ?: d
    fun long(k: String, d: Long) = m[k]?.toLong() ?: d
    fun dbl(k: String, d: Double) = m[k]?.toDouble() ?: d
    fun flag(k: String) = m[k] == "true"
}

private fun readFeatureCsv(f: File): LocalData {
    val xs = ArrayList<DoubleArray>(); val ys = ArrayList<Int>()
    f.forEachLine { line ->
        if (line.isBlank()) return@forEachLine
        val p = line.split(',')
        ys.add(p[0].toInt()); xs.add(DoubleArray(p.size - 1) { p[it + 1].toDouble() })
    }
    return LocalData(xs, ys.toIntArray())
}

private fun phone(a: Args) {
    val id = a.str("client")
    val fs = a.str("features", Constants.DEFAULT_FEATURE_SET)
    val root = File(a.str("root"))
    val train = TrainConfig(a.int("epochs", 2), a.dbl("lr", 0.3), a.int("batch", 32), a.dbl("l2", 0.0), a.dbl("mu", 0.0))
    val cfg = ClientConfig(
        clientId = id, serverUrl = a.str("server"), featureSet = fs, train = train,
        codec = a.str("codec", CODEC_F32), policy = ParticipationPolicy(a.flag("adaptive"), a.dbl("threshold", 0.0)),
        baseSeed = a.long("seed", 0), pollMs = a.long("poll", 200), retryBackoffMs = a.long("backoff", 500),
    )
    val data = Dataset.load(root, id, "train", fs) { System.err.println(it) }
    val tests = linkedMapOf(
        "own_test" to Dataset.load(root, id, "test", fs),
        "newuser" to Dataset.load(root, id, "newuser", fs),
    )
    println("[$id] ${data.size} local training windows, classes=${data.y.toSet().sorted()}")
    val drop = a.str("drop-rounds", "").takeIf { it.isNotEmpty() }?.split('-')?.map { it.toInt() }
    val dropMs = a.long("drop-ms", 0)
    val client = FlClient(cfg, data, log = { println("[$id] $it") }, beforeUpload = { r ->
        if (drop != null && r in drop[0]..drop[1]) { println("[$id] simulating dropout in round $r for ${dropMs} ms"); Thread.sleep(dropMs) }
    })
    val info = client.register()
    val rounds = info.int("max_rounds")
    val sum = client.runFederated()
    println("[$id] federation done: $sum")

    val local = client.trainLocalOnly(rounds)
    client.uploadLocalModel(local)
    // wait (bounded) until the other phone's local model is on the server, then score all three
    val other = Constants.CLIENT_POSES.keys.first { it != id }
    var waited = 0
    while (client.fetchLocalModel(other) == null && waited < 120) { Thread.sleep(500); waited++ }
    val res = client.evaluateAllAndReport(tests, local)

    a.str("out", "").takeIf { it.isNotEmpty() }?.let { out ->
        File(out).mkdirs()
        File(out, "client_$id.json").writeText(MiniJson.write(linkedMapOf(
            "client" to id, "summary" to sum.toString(),
            "traffic_body_bytes_sent" to sum.bytesSent, "traffic_body_bytes_received" to sum.bytesReceived,
            "global_weights" to (client.fetchGlobal()?.toList()),
            "local_weights" to local.toList(),
            "eval" to res.mapValues { (_, v) -> v.mapValues { it.value.toMap() } },
        )))
    }
}

private fun check(cond: Boolean, msg: String) { if (!cond) { System.err.println("SELFTEST FAIL: $msg"); exitProcess(1) } }

private fun selfTest() {
    // 1. SplitMix64 reference values (Python asserts the same numbers)
    val r = SplitMix64(0)
    check(r.nextU64() == 0xE220A8397B1DCDAFuL.toLong(), "splitmix #1")
    check(r.nextU64() == 0x6E789E6AA1B965F4uL.toLong(), "splitmix #2")
    // 2. JSON round trip
    val obj = linkedMapOf("a" to 1, "b" to listOf(1.5, 2.0, -3.25e-7), "c" to "x\"y\n", "d" to true, "e" to null)
    val back = MiniJson.parseObject(MiniJson.write(obj))
    check(back.int("a") == 1 && (back["b"] as List<*>)[2] == -3.25e-7 && back.str("c") == "x\"y\n" && back.bool("d") && back["e"] == null, "json")
    // 3. codecs
    val rnd = SplitMix64(5)
    val d = DoubleArray(78) { (rnd.nextDouble() - 0.5) * 0.6 }
    val (sc, p) = Wire.encodeInt8(d)
    val dec = Wire.decodeInt8(sc, p)
    check(d.indices.all { Math.abs(d[it] - dec[it]) <= Math.abs(d.maxOf { v -> Math.abs(v) }) / 127 / 2 + 1e-9 }, "int8 error bound")
    check(Wire.b64ToF32(Wire.f32ToB64(d)).indices.all { Math.abs(Wire.b64ToF32(Wire.f32ToB64(d))[it] - d[it]) < 1e-6 }, "f32 roundtrip")
    check(Wire.decodeInt8(1.0, Wire.encodeInt8(DoubleArray(78)).second).all { it == 0.0 }, "int8 zero")
    // 4. learning on separable synthetic features
    val nf = 4
    val xs = ArrayList<DoubleArray>(); val ys = ArrayList<Int>()
    val g = SplitMix64(1)
    repeat(300) {
        val x = DoubleArray(nf) { (g.nextDouble() - 0.5) * 4 }
        xs.add(x); ys.add((if (x[0] > 0) 1 else 0) + (if (x[1] > 0) 2 else 0))
    }
    val data = LocalData(xs, ys.toIntArray())
    val m = SoftmaxModel(nf)
    val w = m.train(SoftmaxModel.initWeights(nf), data, TrainConfig(epochs = 60, lr = 0.5), 3)
    check(m.lossAcc(w, data).second > 0.9, "sgd learns")
    // 5. live predictor sees a constant gravity direction as the right pose (portrait_up: +y)
    val fs = "acc_grav"
    val nfs = Constants.nFeatures(fs)
    val wm = DoubleArray(Constants.nParams(nfs))
    // hand built classifier: logit_k = 10 * (gravity mean along the axis pointing up for pose k)
    val up = arrayOf(intArrayOf(2, 1), intArrayOf(2, -1), intArrayOf(1, 1), intArrayOf(1, -1), intArrayOf(0, -1), intArrayOf(0, 1))
    for (k in 0 until 6) wm[k * nfs + 6 + up[k][0]] = 10.0 * up[k][1]  // grav mean x,y,z are features 6,7,8
    val lp = LivePredictor(fs, wm)
    for (i in 0 until 120) {
        val t = 1_000_000_000L + i * 20_000_000L
        lp.push("acc", t, 0.1, 9.7, 0.2); lp.push("grav", t, 0.0, 9.81, 0.0)
    }
    val pr = lp.predict()
    check(pr != null && pr.label == 2, "live predictor portrait_up, got ${pr?.label}")
    // 6. blind test path: buffer -> mark -> 2 s clip -> label (portrait_up gravity), using the hand built classifier above
    val buf2 = SessionBuffer()  // the first 1.5 s are the 'countdown' with the WRONG pose (screen up)
    var mark: Map<String, Int> = emptyMap()
    for (i in 0 until 200) {
        val t = 5_000_000_000L + i * 20_000_000L
        val before = i < 75
        buf2.push("acc", t, 0.0, if (before) 0.0 else 9.7, if (before) 9.8 else 0.1)
        buf2.push("grav", t, 0.0, if (before) 0.0 else 9.81, if (before) 9.81 else 0.0)
        if (i == 74) mark = buf2.mark()
    }
    val clip = buf2.sessionSince(mark)
    val t0 = BlindTest.clipStart(clip, fs)
    check(t0 != null && clip.getValue("acc").size == 125, "mark/sessionSince keeps only post-countdown samples")
    val cw = BlindTest.clipWindows(clip, fs, t0!!)
    val pp = BlindTest.clipProba(SoftmaxModel(nfs), wm, cw)
    check(cw.size == 2 && pp != null && BlindTest.argmax(pp) == 2 && Math.abs(pp.sum() - 1.0) < 1e-12, "blind clip label portrait_up, got ${pp?.toList()}")
    println("SELFTEST OK")
}
