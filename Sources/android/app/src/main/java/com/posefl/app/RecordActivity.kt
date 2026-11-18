package com.posefl.app

import android.app.Activity
import android.app.AlertDialog
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.View
import android.view.WindowManager
import android.widget.AdapterView
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.RadioGroup
import android.widget.Spinner
import android.widget.TextView
import com.posefl.core.FeatureExtractor
import com.posefl.core.LabelCheck
import com.posefl.core.Pose
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/**
 * Guided recording of one labelled session: choose split + pose, press Start, 3-beep countdown,
 * record for N seconds, then a label check and Keep / Discard.
 */
class RecordActivity : Activity() {
    private enum class State { IDLE, COUNTDOWN, RECORDING, SAVING }

    private lateinit var prefs: AppPrefs
    private lateinit var recorder: SensorRecorder
    private lateinit var store: SessionStore

    private lateinit var splitGroup: RadioGroup
    private lateinit var poseSpinner: Spinner
    private lateinit var poseDesc: TextView
    private lateinit var holdingSpinner: Spinner
    private lateinit var durationSpinner: Spinner
    private lateinit var livePose: TextView
    private lateinit var liveGravity: TextView
    private lateinit var status: TextView
    private lateinit var counts: TextView
    private lateinit var startButton: Button
    private lateinit var cancelButton: Button

    private val ui = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()
    private var tone: ToneGenerator? = null

    private var poses: List<Pose> = Pose.entries.toList()
    private var state = State.IDLE
    private var countdown = 0
    private var recordStartMs = 0L
    private var recordDurationMs = 20_000L
    private var recordStartedAt = 0L

    private val holdingKeys = listOf("hand", "surface", "propped")
    private val holdingLabels = listOf("In hand", "On a surface", "Propped / against something")
    private val durationsS = listOf(10, 20, 30, 60)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_record)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        prefs = AppPrefs(this)
        recorder = SensorRecorder(this)
        store = SessionStore(this)
        tone = try { ToneGenerator(AudioManager.STREAM_MUSIC, 90) } catch (e: RuntimeException) { null }

        splitGroup = findViewById(R.id.split_group)
        poseSpinner = findViewById(R.id.pose)
        poseDesc = findViewById(R.id.pose_desc)
        holdingSpinner = findViewById(R.id.holding)
        durationSpinner = findViewById(R.id.duration)
        livePose = findViewById(R.id.live_pose)
        liveGravity = findViewById(R.id.live_gravity)
        status = findViewById(R.id.status)
        counts = findViewById(R.id.counts)
        startButton = findViewById(R.id.start)
        cancelButton = findViewById(R.id.cancel)

        findViewById<TextView>(R.id.who).text =
            "Client ${prefs.clientId} · participant ${prefs.participant} · ${Build.MODEL}"

        holdingSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, holdingLabels)
        durationSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, durationsS.map { "$it s" })
        durationSpinner.setSelection(1)

        splitGroup.setOnCheckedChangeListener { _, _ -> refreshPoses() }
        poseSpinner.setOnItemSelectedListener(object : AdapterView.OnItemSelectedListener {
            override fun onItemSelected(parent: AdapterView<*>?, view: View?, position: Int, id: Long) {
                poseDesc.text = selectedPose().description
                refreshCounts()
            }

            override fun onNothingSelected(parent: AdapterView<*>?) {}
        })
        refreshPoses()

        startButton.setOnClickListener { startCountdown() }
        cancelButton.setOnClickListener { cancelRecording("Cancelled") }
    }

    override fun onResume() {
        super.onResume()
        recorder.start()
        ui.post(liveTick)
    }

    override fun onPause() {
        super.onPause()
        if (state == State.COUNTDOWN || state == State.RECORDING) cancelRecording("Cancelled (app left the screen)")
        recorder.stop()
        ui.removeCallbacks(liveTick)
    }

    override fun onDestroy() {
        super.onDestroy()
        tone?.release()
        io.shutdown()
    }

    // ---------------------------------------------------------------- selections

    private fun selectedSplit(): String = when (splitGroup.checkedRadioButtonId) {
        R.id.split_val -> "val"
        R.id.split_test -> "test"
        else -> "train"
    }

    private fun selectedPose(): Pose = poses[poseSpinner.selectedItemPosition.coerceIn(0, poses.size - 1)]

    /** Training sessions are restricted to this client's three poses (the non-IID split). */
    private fun refreshPoses() {
        val previous = if (poseSpinner.adapter != null) selectedPose() else null
        poses = if (selectedSplit() == "train") Pose.trainPosesFor(prefs.clientId) else Pose.entries.toList()
        poseSpinner.adapter = ArrayAdapter(this, android.R.layout.simple_spinner_dropdown_item, poses.map { it.label })
        val idx = poses.indexOf(previous)
        poseSpinner.setSelection(if (idx >= 0) idx else 0)
        poseDesc.text = selectedPose().description
        refreshCounts()
    }

    private fun refreshCounts() {
        val pose = selectedPose()
        val split = selectedSplit()
        io.execute {
            val mine = store.list().filter { it.client == prefs.clientId && it.pose == pose && it.split == split }
            val text = "Already saved for ${pose.label} / $split: ${mine.size} sessions, ${mine.sumOf { it.windows }} windows"
            ui.post { counts.text = text }
        }
    }

    private fun setControlsEnabled(enabled: Boolean) {
        for (i in 0 until splitGroup.childCount) splitGroup.getChildAt(i).isEnabled = enabled
        poseSpinner.isEnabled = enabled
        holdingSpinner.isEnabled = enabled
        durationSpinner.isEnabled = enabled
        startButton.isEnabled = enabled
        cancelButton.isEnabled = !enabled
    }

    // ---------------------------------------------------------------- live label check

    private val liveTick = object : Runnable {
        override fun run() {
            val g = recorder.latestGravity()
            val guess = Pose.guessFromGravity(g[0], g[1], g[2])
            val target = selectedPose()
            livePose.text = "Phone looks like: " + (guess?.label ?: "between poses")
            livePose.setTextColor(getColor(when (guess) {
                target -> R.color.ok
                null -> R.color.warn
                else -> R.color.bad
            }))
            liveGravity.text = String.format(Locale.US, "gravity  x=%6.2f  y=%6.2f  z=%6.2f  m/s²", g[0], g[1], g[2])
            if (state == State.RECORDING) {
                val left = (recordStartMs + recordDurationMs - System.currentTimeMillis()) / 1000.0
                status.text = String.format(Locale.US, "Recording… %.0f s", maxOf(0.0, left))
            }
            ui.postDelayed(this, 150)
        }
    }

    // ---------------------------------------------------------------- recording flow

    private fun startCountdown() {
        if (state != State.IDLE) return
        state = State.COUNTDOWN
        setControlsEnabled(false)
        countdown = 3
        ui.post(countdownTick)
    }

    private val countdownTick = object : Runnable {
        override fun run() {
            if (state != State.COUNTDOWN) return
            if (countdown > 0) {
                status.text = "Get into position… $countdown"
                beep(ToneGenerator.TONE_PROP_BEEP, 120)
                countdown--
                ui.postDelayed(this, 1000)
            } else {
                beginRecording()
            }
        }
    }

    private fun beginRecording() {
        state = State.RECORDING
        recordDurationMs = durationsS[durationSpinner.selectedItemPosition] * 1000L
        recordStartMs = System.currentTimeMillis()
        recordStartedAt = recordStartMs
        recorder.beginRecording()
        beep(ToneGenerator.TONE_PROP_BEEP, 500)
        vibrate(300)
        ui.postDelayed(finishRecording, recordDurationMs)
    }

    private val finishRecording = Runnable {
        if (state != State.RECORDING) return@Runnable
        val rows = recorder.endRecording()
        state = State.SAVING
        beep(ToneGenerator.TONE_PROP_BEEP2, 300)
        vibrate(150)
        status.text = "Checking…"
        val pose = selectedPose()
        val split = selectedSplit()
        val holding = holdingKeys[holdingSpinner.selectedItemPosition]
        io.execute {
            val windows = FeatureExtractor.windowRanges(rows).size
            val match = LabelCheck.matchFraction(rows, pose)
            ui.post { showResult(rows, pose, split, holding, windows, match) }
        }
    }

    private fun cancelRecording(message: String) {
        ui.removeCallbacks(countdownTick)
        ui.removeCallbacks(finishRecording)
        recorder.endRecording()
        state = State.IDLE
        status.text = message
        setControlsEnabled(true)
    }

    private fun showResult(rows: List<DoubleArray>, pose: Pose, split: String, holding: String, windows: Int, match: Double) {
        val seconds = recordDurationMs / 1000.0
        val rate = rows.size / seconds
        val msg = String.format(Locale.US,
            "%s · %s · %s\n\nSamples: %d (%.0f per second)\nUsable 1-second windows: %d\n" +
                "Label check: %.0f%% of samples look like %s",
            pose.label, split, holding, rows.size, rate, windows, match * 100, pose.label)
        val builder = AlertDialog.Builder(this).setMessage(msg).setCancelable(false)
        if (windows == 0) {
            builder.setTitle("Nothing usable – recording too short")
                .setPositiveButton("OK") { _, _ -> finishSaving("Discarded") }
        } else {
            builder.setTitle(if (match < 0.8) "⚠ This may have the wrong label" else "Recording finished")
                .setPositiveButton("Keep") { _, _ -> save(rows, pose, split, holding, windows, match) }
                .setNegativeButton("Discard") { _, _ -> finishSaving("Discarded") }
        }
        builder.show()
    }

    private fun save(rows: List<DoubleArray>, pose: Pose, split: String, holding: String, windows: Int, match: Double) {
        status.text = "Saving…"
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date(recordStartedAt))
        val meta = SessionMeta(
            id = "${prefs.clientId}_${prefs.participant}_${pose.key}_${split}_$stamp",
            client = prefs.clientId, participant = prefs.participant, pose = pose, split = split,
            holding = holding, startedAt = recordStartedAt, durationMs = recordDurationMs,
            samples = rows.size, windows = windows, labelMatch = match,
            device = "${Build.MANUFACTURER} ${Build.MODEL}", sensors = recorder.availableSensors(),
        )
        io.execute {
            val result = try {
                store.save(meta, rows)
                "Saved ✓ ($windows windows)"
            } catch (e: Exception) {
                "Save failed: ${e.message}"
            }
            ui.post { finishSaving(result) }
        }
    }

    private fun finishSaving(message: String) {
        state = State.IDLE
        status.text = message
        setControlsEnabled(true)
        refreshCounts()
    }

    // ---------------------------------------------------------------- feedback (works with the screen face-down)

    private fun beep(toneType: Int, ms: Int) {
        try { tone?.startTone(toneType, ms) } catch (e: RuntimeException) { /* audio unavailable */ }
    }

    private fun vibrate(ms: Long) {
        val v = getSystemService(Vibrator::class.java) ?: return
        if (v.hasVibrator()) v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
    }
}
