// End-to-end test of the phone's federated-learning client (core/FlClient.kt) against the real server.
// It runs the exact Kotlin code the app uses, as two simulated phones with synthetic sessions.
// Usage: FlEndToEnd <server url> <sessions dir from make_sessions.py> [follow <ms>] [flaky]
//   follow: keep training as the server starts new runs (sweeps), stop after <ms>
//   flaky:  phone B loses the network from 1.5 s to 6.5 s after start (dropout recovery test)
import com.posefl.core.EvalRow
import com.posefl.core.FlListener
import com.posefl.core.FlRunner
import com.posefl.core.HttpTransport
import com.posefl.core.Json
import com.posefl.core.LabeledSession
import com.posefl.core.LiveTest
import com.posefl.core.LivePredictor
import com.posefl.core.LocalData
import com.posefl.core.ModelStore
import com.posefl.core.Pose
import com.posefl.core.SensorCsv
import com.posefl.core.Transport
import com.posefl.core.list
import com.posefl.core.str
import java.io.File
import kotlin.concurrent.thread
import kotlin.system.exitProcess

fun loadSessions(dir: File): List<LabeledSession> =
    Json.parse(File(dir, "sessions.json").readText()).let { it as List<*> }.map {
        @Suppress("UNCHECKED_CAST") val m = it as Map<String, Any?>
        LabeledSession(m.str("id"), Pose.fromKey(m.str("pose")), m.str("split"), m.str("participant"),
            SensorCsv.parse(File(dir, m.str("id") + ".csv").readText()))
    }

fun main(args: Array<String>) {
    val url = args[0]
    val root = File(args[1])
    val follow = args.size > 2 && args[2] == "follow"
    val flaky = "flaky" in args
    val start = System.currentTimeMillis()
    val results = HashMap<String, MutableList<List<EvalRow>>>()
    val runners = ArrayList<FlRunner>()
    val threads = listOf("A", "B").map { cid ->
        val sessions = loadSessions(File(root, cid))
        val cache = HashMap<String, LocalData>()
        val store = ModelStore(File(root, "models_$cid"))
        val http = HttpTransport(url)
        val transport = if (flaky && cid == "B") object : Transport {
            override fun request(method: String, path: String, body: Map<String, Any?>?): Map<String, Any?> {
                val t = System.currentTimeMillis() - start
                if (t in 1500..6500) throw java.net.ConnectException("simulated Wi-Fi loss")
                return http.request(method, path, body)
            }
        } else http
        val runner = FlRunner(transport, cid, "jvm-test", if (cid == "A") "alice" else "bob",
            { spec -> cache.getOrPut(spec) { LocalData.build(sessions, cid, spec) } }, store,
            object : FlListener {
                override fun log(message: String) = println("[$cid] $message")
                override fun progress(runId: String, version: Int, rounds: Int, phase: String, lastValAcc: Double?) {}
                override fun results(runId: String, featureSet: String, rows: List<EvalRow>) {
                    synchronized(results) { results.getOrPut(cid) { ArrayList() }.add(rows) }
                }
            })
        runners.add(runner)
        thread { runner.run(follow) }
    }
    if (follow) {
        // stop after the server has handed out runs for a while (sweep test)
        val deadline = System.currentTimeMillis() + (args.getOrNull(3)?.toLong() ?: 60_000L)
        while (System.currentTimeMillis() < deadline) Thread.sleep(500)
        runners.forEach { it.stopRequested = true }
    }
    threads.forEach { it.join(180_000) }

    var ok = true
    for (cid in listOf("A", "B")) {
        val runs = results[cid] ?: run { println("FAIL no results from $cid"); ok = false; emptyList() }
        for (rows in runs) for (r in rows.filter { it.testSet != "val" }) {
            println(String.format("%s %-12s %-8s acc %.3f  macroF1 %.3f  n=%d", cid, r.testSet, r.modelName, r.metrics.accuracy, r.metrics.macroF1, r.metrics.total))
        }
        val last = runs.lastOrNull() ?: continue
        val g = last.first { it.modelName == "global" && it.testSet == "test" }.metrics.accuracy
        val l = last.first { it.modelName == "local_$cid" && it.testSet == "test" }.metrics.accuracy
        if (g < 0.9 || l > 0.6) { println("FAIL $cid global=$g local=$l"); ok = false }
    }

    // live predictor + test mode on a held-out session, using the models the runner saved
    val models = ModelStore(File(root, "models_A")).loadAll()
    println("saved models on phone A: " + models.joinToString { "${it.name}(${it.featureSet})" })
    if (models.size != 3) { println("FAIL expected 3 saved models"); ok = false }
    val predictor = LivePredictor()
    val test = LiveTest(models.map { it.name }, seed = 1)
    for (s in loadSessions(File(root, "A")).filter { it.split == "test" && it.participant == "carol" }) {
        predictor.reset()
        var frame = predictor.update(emptyList(), models)
        // feed the session in 250 ms steps like the phone's screen does
        var t = s.rows.first()[0] + 1500
        while (t < s.rows.last()[0] - 1000) {
            frame = predictor.update(s.rows.filter { it[0] <= t && it[0] > t - 3000 }, models)
            t += 250
        }
        test.record(s.pose, frame.predictions.associate { it.modelName to it.best }, "carol")
    }
    println("live test (unseen user carol): " + test.summary())
    if (test.accuracy("global") < 0.99) { println("FAIL live global accuracy"); ok = false }
    println(if (ok) "END-TO-END PASSED" else "END-TO-END FAILED")
    exitProcess(if (ok) 0 else 1)
}
