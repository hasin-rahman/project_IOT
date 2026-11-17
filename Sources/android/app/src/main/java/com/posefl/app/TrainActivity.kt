package com.posefl.app

import android.app.Activity
import android.app.AlertDialog
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.WindowManager
import android.widget.Button
import android.widget.CheckBox
import android.widget.ScrollView
import android.widget.TextView
import com.posefl.core.EvalRow
import com.posefl.core.FeatureSets
import com.posefl.core.FlListener
import com.posefl.core.FlRunner
import com.posefl.core.HttpTransport
import com.posefl.core.LabeledSession
import com.posefl.core.LocalData
import com.posefl.core.ModelStore
import com.posefl.core.Pose
import java.io.File
import java.util.Locale

/**
 * Federated training on this phone. Press Start: the phone trains its local-only baseline, then joins
 * the federated rounds the server runs, and finally evaluates all three models on its own validation /
 * test sessions. Raw recordings and features never leave the phone (see core/FlClient.kt).
 */
class TrainActivity : Activity() {
    private lateinit var prefs: AppPrefs
    private lateinit var store: SessionStore
    private lateinit var models: ModelStore

    private lateinit var dataSummary: TextView
    private lateinit var phaseText: TextView
    private lateinit var roundText: TextView
    private lateinit var resultsText: TextView
    private lateinit var logText: TextView
    private lateinit var follow: CheckBox
    private lateinit var startButton: Button
    private lateinit var stopButton: Button

    private val ui = Handler(Looper.getMainLooper())
    private val logLines = ArrayDeque<String>()
    /** The exact JSON of the most recent upload to each endpoint (shown by "Show what this phone uploaded"). */
    private val lastUploads = java.util.concurrent.ConcurrentHashMap<String, String>()
    @Volatile private var runner: FlRunner? = null
    private var worker: Thread? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "Federated training"
        prefs = AppPrefs(this)
        store = SessionStore(this)
        models = ModelStore(File(filesDir, "models"))

        val root = vbox(padding = 16)
        root.add(label("Client ${prefs.clientId} · ${Build.MODEL} · server ${prefs.serverUrl}", 13f, colorRes = R.color.muted))
        dataSummary = label("Loading recorded sessions…", 13f, mono = true).apply {
            background = card(R.color.panel, 8)
            setPadding(dp(10), dp(10), dp(10), dp(10))
        }
        root.add(dataSummary, dp(8))

        follow = CheckBox(this).apply {
            text = "Keep following new server runs (for the sensor study / seed sweeps)"
            isChecked = prefs.followRuns
            setOnCheckedChangeListener { _, checked -> prefs.followRuns = checked }
        }
        root.add(follow, dp(8))

        val buttons = hbox()
        startButton = button("Start") { start() }
        stopButton = button("Stop") { stop() }.apply { isEnabled = false }
        buttons.add(startButton, weight = 1f).add(stopButton, weight = 1f)
        root.add(buttons, dp(4))
        root.add(button("Show what this phone uploaded") { showUploads() }, dp(2))

        phaseText = label("Not started", 22f, bold = true)
        roundText = label("", 15f, colorRes = R.color.muted)
        root.add(phaseText, dp(12)).add(roundText)

        resultsText = label("", 13f, mono = true)
        root.add(resultsText, dp(8))

        root.add(label("Log", 13f, bold = true, colorRes = R.color.muted), dp(12))
        logText = label("", 11f, mono = true).apply {
            background = card(R.color.panel, 8)
            setPadding(dp(8), dp(8), dp(8), dp(8))
        }
        root.add(logText, dp(4))
        // The whole screen scrolls, so the full results table and the log can be reached on any screen size.
        setContentView(ScrollView(this).apply { addView(root) })

        refreshSummary()
    }

    override fun onDestroy() {
        super.onDestroy()
        runner?.stopRequested = true
    }

    private fun refreshSummary() {
        val client = prefs.clientId
        Thread {
            val all = store.list().filter { it.client == client }
            val sb = StringBuilder(String.format(Locale.US, "%-16s %6s %6s %6s\n", "windows", "train", "val", "test"))
            for (p in Pose.entries) {
                val cell = { split: String -> all.filter { it.pose == p && it.split == split }.sumOf { it.windows } }
                val train = if (p in Pose.trainPosesFor(client)) cell("train").toString() else "-"
                sb.append(String.format(Locale.US, "%-16s %6s %6d %6d\n", p.label, train, cell("val"), cell("test")))
            }
            val people = all.filter { it.split == "train" }.map { it.participant }.toSet()
            val testPeople = all.filter { it.split == "test" }.map { it.participant }.toSet()
            sb.append("training participants: ${people.joinToString().ifEmpty { "none" }}")
            val unseen = testPeople - people
            if (unseen.isNotEmpty()) sb.append("\nunseen-user test participants: ${unseen.joinToString()}")
            ui.post { dataSummary.text = sb.toString() }
        }.start()
    }

    private fun start() {
        if (worker?.isAlive == true) return
        val url = prefs.serverUrl
        val transport = try { HttpTransport(url) } catch (e: IllegalArgumentException) {
            appendLog(e.message ?: "bad server URL"); return
        }
        startButton.isEnabled = false
        stopButton.isEnabled = true
        follow.isEnabled = false
        resultsText.text = ""
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        val client = prefs.clientId
        val followRuns = follow.isChecked

        worker = Thread {
            appendLog("Loading sessions…")
            val sessions: List<LabeledSession> = store.loadLabeled(client)
            val cache = HashMap<String, LocalData>()
            val r = FlRunner(
                transport, client, "${Build.MANUFACTURER} ${Build.MODEL}", prefs.participant,
                { spec -> cache.getOrPut(spec) { LocalData.build(sessions, client, spec) } },
                models, listener,
            )
            runner = r
            try {
                r.run(followRuns)
            } catch (e: Exception) {
                appendLog("Stopped with an error: ${e.javaClass.simpleName}: ${e.message}")
                ui.post { phaseText.text = "Error"; phaseText.setTextColor(getColor(R.color.bad)) }
            }
            ui.post {
                startButton.isEnabled = true
                stopButton.isEnabled = false
                follow.isEnabled = true
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }.also { it.start() }
    }

    private fun stop() {
        runner?.stopRequested = true
        stopButton.isEnabled = false
        appendLog("Stopping after the current step…")
    }

    private val listener = object : FlListener {
        override fun log(message: String) = appendLog(message)

        override fun uploaded(endpoint: String, json: String) { lastUploads[endpoint] = json }

        override fun progress(runId: String, version: Int, rounds: Int, phase: String, lastValAcc: Double?) {
            ui.post {
                val (text, color) = when (phase) {
                    "connecting" -> "Connecting to server…" to R.color.warn
                    "baseline" -> "Training local-only model" to R.color.primary
                    "training" -> "Federated training" to R.color.primary
                    "waiting" -> "Waiting for the other phone" to R.color.warn
                    "evaluating" -> "Evaluating the three models" to R.color.primary
                    "done", "idle" -> "Done" to R.color.ok
                    else -> "Error" to R.color.bad
                }
                phaseText.text = text
                phaseText.setTextColor(getColor(color))
                if (rounds > 0) {
                    roundText.text = "Round ${minOf(version, rounds)} of $rounds" +
                        (lastValAcc?.let { String.format(Locale.US, " · global model on my validation data: %.1f%%", it * 100) } ?: "") +
                        "\nrun $runId"
                }
            }
        }

        override fun results(runId: String, featureSet: String, rows: List<EvalRow>) {
            val sb = StringBuilder("Results (run $runId, features $featureSet, ${FeatureSets.size(featureSet)} features)\n")
            for (set in rows.map { it.testSet }.distinct()) {
                sb.append("\n$set\n")
                sb.append(String.format(Locale.US, "  %-12s %8s %9s\n", "model", "accuracy", "macro-F1"))
                for (r in rows.filter { it.testSet == set }) {
                    sb.append(String.format(Locale.US, "  %-12s %7.1f%% %9.3f\n", r.modelName, r.metrics.accuracy * 100, r.metrics.macroF1))
                }
                // the key comparison: a local-only model works on its own three poses and fails on the other three
                sb.append(String.format(Locale.US, "  %-12s %8s %9s\n", "", "A poses", "B poses"))
                for (r in rows.filter { it.testSet == set }) {
                    sb.append(String.format(Locale.US, "  %-12s %8s %9s\n", r.modelName,
                        percent(r.metrics.accuracyOn(Pose.trainPosesFor("A"))), percent(r.metrics.accuracyOn(Pose.trainPosesFor("B")))))
                }
                rows.firstOrNull { it.testSet == set && it.modelName == "global" }?.let { g ->
                    sb.append("  global recall: " + Pose.entries.joinToString(" ") {
                        String.format(Locale.US, "%s %.0f%%", short(it), g.metrics.perClassRecall[it.index] * 100)
                    } + "\n")
                }
            }
            ui.post { resultsText.text = sb.toString() }
        }
    }

    private fun percent(v: Double?): String = if (v == null) "-" else String.format(Locale.US, "%.1f%%", v * 100)

    /**
     * Privacy evidence: shows the exact JSON this phone last sent to the server. It contains the
     * model weights (W, b), the sample count, the per-pose counts and the loss. No sensor readings.
     */
    private fun showUploads() {
        val text = if (lastUploads.isEmpty()) "Nothing uploaded yet. Press Start first."
        else listOf("/update", "/local_model", "/metrics", "/register", "/done")
            .mapNotNull { ep -> lastUploads[ep]?.let { "POST $ep  (${it.length} bytes)\n$it" } }
            .joinToString("\n\n")
        val view = label(text, 11f, mono = true).apply { setPadding(dp(16), dp(8), dp(16), dp(8)); setTextIsSelectable(true) }
        AlertDialog.Builder(this).setTitle("Last upload to each endpoint")
            .setView(ScrollView(this).apply { addView(view) })
            .setPositiveButton("Close", null).show()
    }

    private fun short(p: Pose) = when (p) {
        Pose.SCREEN_UP -> "SU"; Pose.SCREEN_DOWN -> "SD"; Pose.PORTRAIT_UP -> "PU"
        Pose.PORTRAIT_DOWN -> "PD"; Pose.LANDSCAPE_LEFT -> "LL"; Pose.LANDSCAPE_RIGHT -> "LR"
    }

    private fun appendLog(message: String) {
        val line = String.format(Locale.US, "%tT  %s", System.currentTimeMillis(), message)
        ui.post {
            logLines.addLast(line)
            while (logLines.size > 200) logLines.removeFirst()
            logText.text = logLines.joinToString("\n")
        }
    }
}
