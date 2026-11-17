package com.fedpose.core

import java.io.IOException

data class ClientConfig(
    val clientId: String,
    val serverUrl: String,
    val featureSet: String = Constants.DEFAULT_FEATURE_SET,
    val train: TrainConfig = TrainConfig(),
    val codec: String = CODEC_F32,
    val policy: ParticipationPolicy = ParticipationPolicy(),
    val baseSeed: Long = 0,
    val pollMs: Long = 1000,
    val retryBackoffMs: Long = 1500,
)

class RunSummary {
    var rounds = 0            // rounds in which this client's upload (or skip) was accepted
    var skipped = 0           // uploads replaced by a skip notice
    var stale = 0             // uploads rejected as stale (re-synced afterwards)
    var offlineRetries = 0    // network failures that were retried
    var bytesSent = 0L
    var bytesReceived = 0L
    var lastVersion = 0
    override fun toString() = "rounds=$rounds skipped=$skipped stale=$stale offlineRetries=$offlineRetries " +
        "bodyBytesSent=$bytesSent bodyBytesReceived=$bytesReceived lastVersion=$lastVersion"
}

fun clientSeed(base: Long, round: Int, clientId: String): Long =
    base * 1_000_003L + round * 1009L + Constants.CLIENT_INDEX.getValue(clientId)

fun localBaselineSeed(base: Long, clientId: String): Long =
    base * 1_000_003L + 777L + Constants.CLIENT_INDEX.getValue(clientId)

/**
 * The phone side of federated learning. All raw sensor data and extracted windows stay in `data`;
 * only model deltas, sample counts, loss/accuracy metadata, and evaluation results go to the server.
 */
class FlClient(
    val cfg: ClientConfig,
    val data: LocalData,
    private val log: (String) -> Unit = {},
    private val shouldStop: () -> Boolean = { false },
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
    /** Test/demo hook called after local training, right before the upload (e.g. to fake a dropout). */
    private val beforeUpload: (round: Int) -> Unit = {},
) {
    val nf = Constants.nFeatures(cfg.featureSet)
    val model = SoftmaxModel(nf)
    val http = FlHttp(cfg.serverUrl)

    /** The most recent upload exactly as it went over the wire, with the payload shortened, for the privacy demo. */
    @Volatile var lastUploadSummary: String = "(nothing uploaded yet)"
        private set

    /** Shows what a body contains without dumping bytes: every field name, payload size and a preview. */
    private fun summarize(body: Map<String, Any?>): String {
        val shown = LinkedHashMap(body)
        (body["payload"] as? String)?.let { shown["payload"] = "<${it.length} base64 chars> ${it.take(12)}..." }
        return MiniJson.write(shown)
    }

    private fun backoff() = sleeper(cfg.retryBackoffMs)

    fun register(): Map<String, Any?> {
        while (true) {
            try {
                val r = http.post("/register", mapOf("client_id" to cfg.clientId))
                check(r.code == 200) { "register failed: ${r.body}" }
                check(r.body.int("n_features") == nf) {
                    "server expects ${r.body.int("n_features")} features, this client uses $nf (${cfg.featureSet})"
                }
                return r.body
            } catch (e: IOException) {
                log("cannot reach server (${e.message}); retrying"); backoff()
            }
        }
    }

    /** Local-only baseline: from scratch, same total epochs a client gets during federation. */
    fun trainLocalOnly(rounds: Int): DoubleArray {
        val c = cfg.train.copy(epochs = rounds * cfg.train.epochs, mu = 0.0)
        return model.train(SoftmaxModel.initWeights(nf), data, c, localBaselineSeed(cfg.baseSeed, cfg.clientId))
    }

    fun runFederated(): RunSummary {
        val sum = RunSummary()
        var lastDoneRound = -1
        var pending: Triple<Int, Int, Map<String, Any?>>? = null  // (round, version, body) kept across outages
        while (!shouldStop()) {
            val m = try {
                http.get("/model?client_id=${cfg.clientId}").body
            } catch (e: IOException) {
                sum.offlineRetries++; log("server unreachable (${e.message}); retrying"); backoff(); continue
            }
            when (m.str("status")) {
                "idle" -> { sleeper(cfg.pollMs); continue }
                "done" -> { log("training finished (server status done)"); break }
            }
            val round = m.int("round")
            val version = m.int("version")
            sum.lastVersion = version
            if (round == lastDoneRound || m.bool("already_submitted")) {
                lastDoneRound = round; sleeper(cfg.pollMs); continue
            }
            val body: Map<String, Any?> =
                if (pending != null && pending.first == round && pending.second == version) pending.third
                else buildBody(m, round, version).also {
                    pending = Triple(round, version, it)
                    lastUploadSummary = summarize(it)
                    log("UPLOAD " + lastUploadSummary)
                    beforeUpload(round)
                }
            val res = try {
                http.post("/update", body)
            } catch (e: IOException) {
                sum.offlineRetries++; log("upload failed (${e.message}); will retry round $round"); backoff(); continue
            }
            when (res.code) {
                200 -> {
                    lastDoneRound = round; pending = null; sum.rounds++
                    val skipped = body["skipped"] == true
                    if (skipped) sum.skipped++
                    log("round $round: ${if (skipped) "skipped (small update)" else "update sent"}")
                }
                409 -> {
                    sum.stale++; pending = null
                    log("round $round rejected (${res.body["error"]}); server is at round ${res.body["round"]}, re-syncing")
                }
                else -> { log("round $round rejected: ${res.body}"); backoff() }
            }
        }
        sum.bytesSent = http.traffic.bytesSent
        sum.bytesReceived = http.traffic.bytesReceived
        return sum
    }

    private fun buildBody(m: Map<String, Any?>, round: Int, version: Int): Map<String, Any?> {
        val w32 = Wire.b64ToF32(m.str("weights"))
        check(w32.size == model.nParams) { "global model has ${w32.size} params, expected ${model.nParams}" }
        val wl = model.train(w32, data, cfg.train, clientSeed(cfg.baseSeed, round, cfg.clientId))
        val delta = DoubleArray(w32.size) { wl[it] - w32[it] }
        val norm = Wire.l2(delta)
        if (cfg.policy.shouldSkip(norm, m.bool("must_participate"))) {
            return Wire.skipBody(cfg.clientId, round, version, norm)
        }
        val (loss, acc) = model.lossAcc(wl, data)
        val counts = IntArray(Constants.N_CLASSES).also { c -> for (l in data.y) c[l]++ }
        return Wire.updateBody(cfg.clientId, round, version, data.size, delta, cfg.codec,
            mapOf("train_loss" to loss, "train_acc" to acc, "class_counts" to counts.toList()))
    }

    // ----------------------------------------------------- model exchange and evaluation
    fun fetchGlobal(): DoubleArray? = try {
        val r = http.get("/model?client_id=${cfg.clientId}")
        if (r.code == 200) Wire.b64ToF32(r.body.str("weights")) else null
    } catch (e: IOException) { null }

    fun uploadLocalModel(w: DoubleArray) {
        val r = http.post("/local_model", mapOf("client_id" to cfg.clientId, "weights" to Wire.f32ToB64(w)))
        check(r.code == 200) { "local model upload failed: ${r.body}" }
    }

    /** The other phone's local-only model, relayed by the server so this phone can score it. */
    fun fetchLocalModel(owner: String): DoubleArray? = try {
        val r = http.get("/local_model?client_id=$owner&asker=${cfg.clientId}")
        if (r.code == 200) Wire.b64ToF32(r.body.str("weights")) else null
    } catch (e: IOException) { null }

    fun report(modelName: String, testSet: String, ev: Eval) {
        val body = LinkedHashMap<String, Any?>()
        body["client_id"] = cfg.clientId; body["model"] = modelName; body["test_set"] = testSet
        body.putAll(ev.toMap())
        http.post("/report", body)
    }

    /**
     * Scores local_A, local_B and global on THIS phone's test sets and reports only the metrics.
     * `ownLocal` is this phone's own local-only model (never replaced by the global model).
     */
    fun evaluateAllAndReport(testSets: Map<String, LocalData>, ownLocal: DoubleArray): Map<String, Map<String, Eval>> {
        val other = Constants.CLIENT_POSES.keys.first { it != cfg.clientId }
        val models = LinkedHashMap<String, DoubleArray?>()
        models["local_${cfg.clientId}"] = ownLocal
        models["local_$other"] = fetchLocalModel(other)
        models["global"] = fetchGlobal()
        val out = LinkedHashMap<String, Map<String, Eval>>()
        for ((name, w) in models) {
            if (w == null) { log("model $name not available yet; skipped"); continue }
            val per = LinkedHashMap<String, Eval>()
            for ((tsName, ts) in testSets) {
                val ev = Eval.of(model, w, ts)
                per[tsName] = ev
                report(name, tsName, ev)
                log("$name on $tsName: acc=${"%.3f".format(ev.accuracy)} macroF1=${"%.3f".format(ev.macroF1)} (n=${ev.n})")
            }
            out[name] = per
        }
        return out
    }
}
