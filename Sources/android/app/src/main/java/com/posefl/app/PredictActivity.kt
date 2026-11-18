package com.posefl.app

import android.app.Activity
import android.app.AlertDialog
import android.media.ToneGenerator
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import com.posefl.core.BlindTestLog
import com.posefl.core.HttpTransport
import com.posefl.core.LiveFrame
import com.posefl.core.LivePredictor
import com.posefl.core.LiveTest
import com.posefl.core.ModelNames
import com.posefl.core.ModelStore
import com.posefl.core.Pose
import com.posefl.core.StoredModel
import com.posefl.core.toMap
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The demo screen. It always shows the live prediction of all three models (federated global,
 * local A, local B) for the same 1-second sensor window, and it has two test modes:
 *
 *  - BLIND TEST (demo day): a judge puts the phone in a pose from a pose card and taps the big
 *    button. The app waits until the phone is still, records ONE window and shows what each of
 *    the three models predicted. The app does not know the true pose; it numbers and logs the placements.
 *  - PRACTICE: the app asks for a pose, so it knows the right answer and scores the three models
 *    the way the demo-day rubric does. Use it to rehearse.
 *
 * It uses the three models saved on the phone by the Federated training screen, so it works
 * without the server.
 */
class PredictActivity : Activity() {
    /** IDLE -> (tap) -> COUNTDOWN (beeps: time to place the phone) -> WAIT_FOR_STILL -> capture -> IDLE */
    private enum class CaptureState { IDLE, COUNTDOWN, WAIT_FOR_STILL }

    private lateinit var prefs: AppPrefs
    private lateinit var recorder: SensorRecorder
    private lateinit var feedback: Feedback
    private val predictor = LivePredictor()
    private var models: List<StoredModel> = emptyList()

    // live display
    private lateinit var modelInfo: TextView
    private lateinit var stillText: TextView
    private val liveViews = HashMap<String, TextView>()      // model name -> its live line

    // test area
    private lateinit var blindMode: RadioButton
    private lateinit var tester: EditText
    private lateinit var promptText: TextView
    private lateinit var statusText: TextView
    private lateinit var captureButton: Button
    private lateinit var resultTitle: TextView
    private val resultViews = HashMap<String, TextView>()    // model name -> its captured answer
    private lateinit var logText: TextView

    private val ui = Handler(Looper.getMainLooper())
    private var blindTest = BlindTestLog()
    private var practice: LiveTest? = null
    private var state = CaptureState.IDLE
    private var target: Pose? = null          // the pose asked for (practice mode only)
    private var countdown = 0
    private var waitStartedAt = 0L
    private var stillSince = 0L

    // ------------------------------------------------------------------ screen layout

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "Live demo and blind test"
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        prefs = AppPrefs(this)
        recorder = SensorRecorder(this)
        feedback = Feedback(this)

        val root = vbox(padding = 14)
        modelInfo = label("", 12f, colorRes = R.color.muted)
        root.add(modelInfo)

        // ---- live: the three models on the same sensor window ----
        val liveCard = whiteCard()
        liveCard.add(sectionTitle("Live · all three models, same 1-second window"))
        for (name in ModelNames.ALL) {
            val line = label("${ModelNames.label(name)}: –", if (name == ModelNames.GLOBAL) 24f else 18f, bold = name == ModelNames.GLOBAL)
            liveViews[name] = line
            liveCard.add(line, dp(4))
        }
        stillText = label("", 14f, bold = true, colorRes = R.color.warn)
        liveCard.add(stillText, dp(2))
        root.add(liveCard, dp(6))

        // ---- test mode selection ----
        val testCard = whiteCard()
        testCard.add(sectionTitle("Test"))
        val modes = RadioGroup(this)
        blindMode = RadioButton(this).apply { text = "Blind test: the judge chooses the pose (demo day)"; id = View.generateViewId() }
        val practiceMode = RadioButton(this).apply { text = "Practice: the app asks for a pose and keeps score"; id = View.generateViewId() }
        modes.addView(blindMode)
        modes.addView(practiceMode)
        modes.check(blindMode.id)
        modes.setOnCheckedChangeListener { _, _ -> showModeChanged() }
        testCard.add(modes)
        tester = EditText(this).apply { hint = "Tester name (practice)"; setText(prefs.participant); setSingleLine() }
        testCard.add(tester, dp(2))

        // ---- capture ----
        promptText = label("", 24f, bold = true).center()
        statusText = label("", 15f, colorRes = R.color.muted).center()
        testCard.add(promptText, dp(8)).add(statusText)
        captureButton = primaryButton("Capture placement") { startCapture() }.apply { textSize = 20f; minimumHeight = dp(72) }
        testCard.add(captureButton, dp(6))
        root.add(testCard, dp(12))

        // ---- the captured answer of each model (stays on screen until the next capture) ----
        val resultCard = whiteCard()
        resultTitle = label("No placement captured yet", 13f, bold = true, colorRes = R.color.primary)
        resultCard.add(resultTitle)
        for (name in ModelNames.ALL) {
            val line = label("${ModelNames.label(name)}: –", 22f, bold = true)
            resultViews[name] = line
            resultCard.add(line, dp(4))
        }
        root.add(resultCard, dp(12))

        val row = hbox()
        row.add(button("Undo last") { undo() }, weight = 1f)
            .add(button("Finish / save") { finishTest() }, weight = 1f)
        root.add(row, dp(4))
        logText = label("", 13f, mono = true)
        root.add(logText, dp(8))

        setContentView(ScrollView(this).apply { addView(root) })
        showModeChanged()
    }

    override fun onResume() {
        super.onResume()
        models = ModelStore(File(filesDir, "models")).loadAll()
        predictor.reset()
        val fmt = SimpleDateFormat("MMM d HH:mm", Locale.US)
        modelInfo.text = if (models.isEmpty()) "No trained models on this phone yet. Run Federated training first."
        else "Models loaded: " + models.joinToString(" · ") { "${ModelNames.label(it.name)} (${it.featureSet}, ${fmt.format(Date(it.savedAt))})" }
        recorder.start()
        ui.post(tick)
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(tick)
        ui.removeCallbacks(countdownTick)
        recorder.stop()
        if (state != CaptureState.IDLE) {
            state = CaptureState.IDLE
            statusText.text = "Capture cancelled"
            captureButton.isEnabled = true
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        feedback.release()
    }

    private fun isBlind(): Boolean = blindMode.isChecked

    // ------------------------------------------------------------------ live display (5 times a second)

    private val tick = object : Runnable {
        override fun run() {
            val frame = predictor.update(recorder.liveRows(), models)
            showLive(frame)
            if (state == CaptureState.WAIT_FOR_STILL) captureWhenStill(frame)
            ui.postDelayed(this, 200)
        }
    }

    private fun showLive(frame: LiveFrame) {
        if (frame.warmingUp) { stillText.text = "Reading sensors…"; return }
        for (name in ModelNames.ALL) {
            val view = liveViews[name] ?: continue
            val p = frame.predictions.firstOrNull { it.modelName == name }
            if (p == null) { view.text = "${ModelNames.label(name)}: no model"; continue }
            view.text = String.format(Locale.US, "%s: %s  %.0f%%", ModelNames.label(name), p.best.label, p.confidence * 100)
            view.setTextColor(getColor(if (p.pose != null) R.color.ok else R.color.warn))
        }
        stillText.text = if (frame.moving) "Moving – hold still" else ""
    }

    // ------------------------------------------------------------------ capturing one placement

    private fun showModeChanged() {
        if (state != CaptureState.IDLE) return
        tester.visibility = if (isBlind()) View.GONE else View.VISIBLE
        target = null
        if (isBlind()) {
            captureButton.text = "Capture placement"
            promptText.setTextColor(getColor(R.color.primary))
            promptText.text = placementPrompt()
            statusText.text = "Tap the button, put the phone in the pose on the card and keep it still until the long beep."
        } else {
            captureButton.text = "Next trial"
            promptText.text = ""
            statusText.text = "Tap Next trial, make the pose shown and keep the phone still until the long beep."
        }
        showLog()
    }

    private fun placementPrompt(): String =
        if (blindTest.nextNumber <= blindTest.plannedPlacements) "Placement ${blindTest.nextNumber} of ${blindTest.plannedPlacements}"
        else "Placement ${blindTest.nextNumber} (all ${blindTest.plannedPlacements} are done)"

    /** The button was tapped: beep three times (time to place the phone), then wait for it to be still. */
    private fun startCapture() {
        if (models.isEmpty()) { statusText.text = "Train first: there are no models on this phone."; return }
        if (state != CaptureState.IDLE) return
        if (isBlind()) {
            target = null
            promptText.text = placementPrompt()
        } else {
            val test = practice ?: LiveTest(models.map { it.name }).also { practice = it }
            val pose = test.nextTarget()
            target = pose
            promptText.text = "Make it: ${pose.label.uppercase(Locale.US)}"
            statusText.text = pose.description
        }
        promptText.setTextColor(getColor(R.color.primary))
        captureButton.isEnabled = false
        state = CaptureState.COUNTDOWN
        countdown = 3
        ui.post(countdownTick)
    }

    private val countdownTick = object : Runnable {
        override fun run() {
            if (state != CaptureState.COUNTDOWN) return
            if (countdown > 0) {
                statusText.text = "Place the phone… $countdown"
                feedback.beep()
                countdown--
                ui.postDelayed(this, 700)
            } else {
                state = CaptureState.WAIT_FOR_STILL
                waitStartedAt = System.currentTimeMillis()
                stillSince = 0L
                statusText.text = "Hold still…"
            }
        }
    }

    /** Captures as soon as the phone has been still for [STILL_MS] (or after [MAX_WAIT_MS] at the latest). */
    private fun captureWhenStill(frame: LiveFrame) {
        if (frame.warmingUp) return
        val now = System.currentTimeMillis()
        if (frame.moving) stillSince = 0L else if (stillSince == 0L) stillSince = now
        val still = stillSince != 0L && now - stillSince >= STILL_MS
        val waitedTooLong = now - waitStartedAt >= MAX_WAIT_MS
        if (still || waitedTooLong) capture(frame, still)
    }

    /**
     * Records each model's answer, shows it and adds it to the log. The answer is averaged over the
     * last two 1-second windows (LivePredictor.captureAnswers); if there is not enough data for that,
     * the newest window alone is used.
     */
    private fun capture(frame: LiveFrame, still: Boolean) {
        val answers = predictor.captureAnswers(recorder.liveRows(), models).ifEmpty { frame.windowAnswers() }
        val pose = target
        val title: String
        if (pose == null) {
            val placement = blindTest.add(answers, still)
            title = "PLACEMENT ${placement.number} – what each model predicted"
            promptText.text = placementPrompt()
        } else {
            val test = practice ?: return
            test.record(pose, answers, tester.text.toString().trim().ifEmpty { "unknown" })
            title = "TRIAL ${test.trials.size} – the pose was ${pose.label}"
            val globalRight = answers[ModelNames.GLOBAL] == pose
            promptText.text = (if (globalRight) "✓ " else "✗ ") + pose.label
            promptText.setTextColor(getColor(if (globalRight) R.color.ok else R.color.bad))
        }
        showCaptured(title + if (still) "" else " (captured while moving)", answers, pose)
        feedback.beep(ToneGenerator.TONE_PROP_BEEP2, 400)
        feedback.vibrate(200)
        state = CaptureState.IDLE
        statusText.text = "Captured. Tap the button for the next one."
        captureButton.isEnabled = true
        showLog()
    }

    /** Fills the result card. In practice mode ([truth] known) each answer gets a ✓ or ✗. */
    private fun showCaptured(title: String, answers: Map<String, Pose>, truth: Pose?) {
        resultTitle.text = title
        for (name in ModelNames.ALL) {
            val view = resultViews[name] ?: continue
            val answer = answers[name]
            val mark = if (truth == null || answer == null) "" else if (answer == truth) "  ✓" else "  ✗"
            view.text = "${ModelNames.label(name)}: ${answer?.label ?: "no model"}$mark"
            view.setTextColor(getColor(when {
                truth == null || answer == null -> R.color.primary
                answer == truth -> R.color.ok
                else -> R.color.bad
            }))
        }
    }

    private fun undo() {
        if (state != CaptureState.IDLE) return
        if (isBlind()) {
            val removed = blindTest.undoLast() ?: return
            statusText.text = "Removed placement ${removed.number}"
            promptText.text = placementPrompt()
        } else {
            val removed = practice?.undoLast() ?: return
            statusText.text = "Removed the last trial (${removed.target.label})"
        }
        showLog()
    }

    // ------------------------------------------------------------------ the log / score under the buttons

    private fun showLog() {
        logText.text = if (isBlind()) blindLogText() else practiceScoreText()
    }

    /** Newest first: "#07  G Landscape Left | A Portrait Up | B Landscape Left". */
    private fun blindLogText(): String {
        if (blindTest.placements.isEmpty()) return ""
        val sb = StringBuilder("#    Global | Local A | Local B\n")
        for (p in blindTest.placements.reversed()) {
            sb.append(String.format(Locale.US, "#%02d  ", p.number))
            sb.append(ModelNames.ALL.joinToString(" | ") { short(p.answers[it]) })
            if (!p.still) sb.append("  (moving)")
            sb.append('\n')
        }
        return sb.toString()
    }

    /** The practice score, laid out like part A of the demo-day rubric (40 points). */
    private fun practiceScoreText(): String {
        val t = practice ?: return ""
        if (t.trials.isEmpty()) return ""
        val posesA = Pose.trainPosesFor("A")
        val posesB = Pose.trainPosesFor("B")
        val sb = StringBuilder("${t.trials.size} trials (the real test has 36)\n")
        sb.append(scoreLine("A1 Global, all poses", t.correctOn(ModelNames.GLOBAL), t.countOn(), 30))
        sb.append(scoreLine("A2 Local A, A poses", t.correctOn(ModelNames.LOCAL_A, posesA), t.countOn(posesA), 5))
        sb.append(scoreLine("A3 Local B, B poses", t.correctOn(ModelNames.LOCAL_B, posesB), t.countOn(posesB), 5))
        sb.append(String.format(Locale.US, "Part A estimate: %.1f / 40\n\n", t.rubricPoints()))
        sb.append("Local models on the OTHER phone's poses (should be about 0):\n")
        sb.append("  Local A on B poses ${t.correctOn(ModelNames.LOCAL_A, posesB)}/${t.countOn(posesB)}")
        sb.append("   Local B on A poses ${t.correctOn(ModelNames.LOCAL_B, posesA)}/${t.countOn(posesA)}\n\n")
        sb.append("Global by pose:\n")
        for (p in Pose.entries) {
            val n = t.countOn(listOf(p))
            if (n > 0) sb.append(String.format(Locale.US, "  %-16s %d/%d\n", p.label, t.correctOn(ModelNames.GLOBAL, listOf(p)), n))
        }
        return sb.toString()
    }

    private fun scoreLine(name: String, correct: Int, count: Int, points: Int): String {
        val accuracy = if (count == 0) 0.0 else correct.toDouble() / count
        return String.format(Locale.US, "%-21s %2d/%-2d -> %4.1f / %d\n", name, correct, count, points * accuracy, points)
    }

    private fun short(p: Pose?) = when (p) {
        Pose.SCREEN_UP -> "ScreenUp"; Pose.SCREEN_DOWN -> "ScreenDn"; Pose.PORTRAIT_UP -> "PortUp"
        Pose.PORTRAIT_DOWN -> "PortDn"; Pose.LANDSCAPE_LEFT -> "LandLeft"; Pose.LANDSCAPE_RIGHT -> "LandRight"
        null -> "-"
    }

    // ------------------------------------------------------------------ finishing a test

    private fun finishTest() {
        if (state != CaptureState.IDLE) return
        if (isBlind()) finishBlindTest() else finishPractice()
    }

    /** Saves the placement log on the phone. Nothing is sent anywhere. */
    private fun finishBlindTest() {
        if (blindTest.placements.isEmpty()) { statusText.text = "No placements to save."; return }
        val file = saveCsv("blind_test", blindTest.toCsv())
        AlertDialog.Builder(this).setTitle("Blind test saved")
            .setMessage("${blindTest.placements.size} placements saved on the phone (${file.name}).")
            .setPositiveButton("Start a new test") { _, _ ->
                blindTest = BlindTestLog()
                resultTitle.text = "No placement captured yet"
                showModeChanged()
            }
            .setNegativeButton("Keep counting", null)
            .show()
    }

    /** Saves the practice trials on the phone and sends the scores (not sensor data) to the server. */
    private fun finishPractice() {
        val t = practice
        if (t == null || t.trials.isEmpty()) { statusText.text = "No trials to save."; return }
        saveCsv("live_test", t.toCsv())
        val names = models.map { it.name }
        val testers = t.trials.map { it.tester }.distinct().joinToString(",")
        val url = prefs.serverUrl
        val password = prefs.serverPassword
        val client = prefs.clientId
        val summary = t.summary() + String.format(Locale.US, "\nPart A estimate: %.1f / 40", t.rubricPoints())
        Thread {
            val msg = try {
                val http = HttpTransport(url, password)
                for (n in names) {
                    http.request("POST", "/metrics", mapOf(
                        "client_id" to client, "model_name" to n, "test_set" to "live_test",
                        "participant" to testers, "metrics" to t.metrics(n).toMap()))
                }
                "Saved on the phone and sent the scores to the server."
            } catch (e: Exception) {
                "Saved on the phone (server not reachable: ${e.message})."
            }
            ui.post {
                AlertDialog.Builder(this).setTitle("Practice finished").setMessage("$summary\n\n$msg")
                    .setPositiveButton("Start a new test") { _, _ -> practice = null; showModeChanged() }
                    .setNegativeButton("Keep counting", null)
                    .show()
            }
        }.start()
    }

    private fun saveCsv(prefix: String, text: String): File {
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        val dir = File(filesDir, "live_tests").apply { mkdirs() }
        return File(dir, "${prefix}_$stamp.csv").also { it.writeText(text) }
    }

    companion object {
        /** The phone must be still this long before a placement is captured. */
        const val STILL_MS = 1000L
        /** If the phone never becomes still, capture anyway after this long. */
        const val MAX_WAIT_MS = 8000L
    }
}
