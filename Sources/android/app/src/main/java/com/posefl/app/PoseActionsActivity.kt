package com.posefl.app

import android.app.Activity
import android.app.NotificationManager
import android.content.Intent
import android.graphics.Typeface
import android.hardware.camera2.CameraAccessException
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.media.ToneGenerator
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.TextView
import com.posefl.core.LivePredictor
import com.posefl.core.ModelNames
import com.posefl.core.ModelStore
import com.posefl.core.PoseAction
import com.posefl.core.PoseActionTrigger
import com.posefl.core.StoredModel
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Pose actions: the IoT loop sense -> decide -> act, entirely on the phone.
 *
 *  SENSE   the sensors fill a 1-second window (SensorRecorder)
 *  DECIDE  the chosen model says which pose it is (LivePredictor); PoseActionTrigger switches an
 *          action on after the pose has been held for 3 s, and off 1 s after the phone leaves it
 *  ACT     Do Not Disturb, desk clock, flashlight or timer (switchOn / switchOff below)
 *
 * The model switch shows what federated learning adds: a local model only knows its own phone's
 * three poses, so it can only trigger two of the four actions; the global model triggers all four.
 * Everything is switched off when the screen is left.
 */
class PoseActionsActivity : Activity() {
    private lateinit var recorder: SensorRecorder
    private lateinit var feedback: Feedback
    private val predictor = LivePredictor()
    private val trigger = PoseActionTrigger()
    private var models: List<StoredModel> = emptyList()
    private var selected = ModelNames.GLOBAL

    private lateinit var modelGroup: RadioGroup
    private val modelButtons = HashMap<String, RadioButton>()
    private lateinit var seesText: TextView
    private lateinit var holdBar: ProgressBar
    private lateinit var holdText: TextView
    private lateinit var activeCard: LinearLayout
    private lateinit var activeTitle: TextView
    private lateinit var activeBig: TextView
    private lateinit var activeHint: TextView
    private val actionRows = HashMap<PoseAction, TextView>()
    private lateinit var dndButton: Button
    private lateinit var logText: TextView
    private val log = ArrayDeque<String>()

    private val ui = Handler(Looper.getMainLooper())
    private val idleBackground by lazy { cardBackground(R.color.card) }
    private val activeBackground by lazy { cardBackground(R.color.active) }
    private var timerStartedAt = 0L
    private var dndBefore: Int? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        title = "Pose actions"
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        recorder = SensorRecorder(this)
        feedback = Feedback(this)

        val root = vbox(padding = 16)
        root.add(label("Hold a pose for 3 seconds and the phone acts on it. Leave the pose for 1 second to switch it off.", 14f,
            colorRes = R.color.muted))

        // ---- which model decides ----
        val modelCard = whiteCard()
        modelCard.add(sectionTitle("Model that decides"))
        modelGroup = RadioGroup(this).apply { orientation = RadioGroup.HORIZONTAL }
        for (name in ModelNames.ALL) {
            val b = RadioButton(this).apply { text = ModelNames.label(name); id = View.generateViewId(); textSize = 16f }
            modelButtons[name] = b
            modelGroup.addView(b, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
        }
        modelGroup.setOnCheckedChangeListener { _, id ->
            val name = modelButtons.entries.firstOrNull { it.value.id == id }?.key ?: return@setOnCheckedChangeListener
            if (name != selected) { selected = name; switchEverythingOff("switched to ${ModelNames.label(name)}") }
        }
        modelCard.add(modelGroup, dp(4))
        seesText = label("–", 22f, bold = true)
        modelCard.add(seesText, dp(8))
        holdBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply { max = 100 }
        modelCard.add(holdBar, dp(6))
        holdText = label("", 13f, colorRes = R.color.muted)
        modelCard.add(holdText)
        root.add(modelCard, dp(12))

        // ---- the action that is on ----
        activeCard = whiteCard(padding = 18)
        activeTitle = label("No action", 24f, bold = true).center()
        activeBig = label("", 56f, bold = true).center().apply { typeface = Typeface.create(Typeface.DEFAULT, Typeface.BOLD) }
        activeHint = label("", 13f, colorRes = R.color.muted).center()
        activeCard.add(activeTitle).add(activeBig).add(activeHint)
        root.add(activeCard, dp(12))

        // ---- the four actions ----
        val listCard = whiteCard()
        listCard.add(sectionTitle("Actions"))
        for (a in PoseAction.entries) {
            val line = label("", 15f)
            actionRows[a] = line
            listCard.add(line, dp(6))
        }
        listCard.add(label("Portrait Up and Landscape Right have no action. A local model knows only its own phone's poses, " +
            "so it can switch on only two of these.", 12f, colorRes = R.color.muted), dp(8))
        dndButton = button("Allow Do Not Disturb access") {
            startActivity(Intent(Settings.ACTION_NOTIFICATION_POLICY_ACCESS_SETTINGS))
        }
        listCard.add(dndButton, dp(6))
        root.add(listCard, dp(12))

        logText = label("", 12f, mono = true, colorRes = R.color.muted)
        root.add(logText, dp(10))
        setContentView(ScrollView(this).apply { addView(root) })
    }

    override fun onResume() {
        super.onResume()
        models = ModelStore(File(filesDir, "models")).loadAll()
        for ((name, b) in modelButtons) b.isEnabled = models.any { it.name == name }
        if (models.none { it.name == selected }) selected = models.firstOrNull()?.name ?: ModelNames.GLOBAL
        modelButtons[selected]?.isChecked = true
        dndButton.visibility = if (hasDndAccess()) View.GONE else View.VISIBLE
        predictor.reset()
        recorder.start()
        showActions()
        ui.post(tick)
    }

    override fun onPause() {
        super.onPause()
        ui.removeCallbacks(tick)
        recorder.stop()
        switchEverythingOff("left the screen")   // nothing stays on when the screen is left
    }

    override fun onDestroy() {
        super.onDestroy()
        feedback.release()
    }

    // ------------------------------------------------------------------ sense + decide (5 times a second)

    private val tick = object : Runnable {
        override fun run() {
            decide()
            showActive()
            ui.postDelayed(this, 200)
        }
    }

    private fun decide() {
        if (models.isEmpty()) { seesText.text = "No models yet: run Federated training first."; return }
        val frame = predictor.update(recorder.liveRows(), models)
        val p = frame.predictions.firstOrNull { it.modelName == selected }
        if (frame.warmingUp || p == null) { seesText.text = "Reading sensors…"; return }
        val now = System.currentTimeMillis()
        seesText.text = String.format(Locale.US, "%s thinks: %s  %.0f%%", ModelNames.label(selected), p.best.label, p.confidence * 100)

        when (val change = trigger.update(p.best, p.confidence, now)) {
            is PoseActionTrigger.Change.Started -> switchOn(change.action)
            is PoseActionTrigger.Change.Stopped -> switchOff(change.action)
            null -> {}
        }
        val holding = trigger.holding
        holdBar.progress = (trigger.holdProgress(now) * 100).toInt()
        holdText.text = when {
            trigger.active != null -> "Leave the pose for 1 s to switch it off"
            holding != null -> String.format(Locale.US, "Holding %s… %s in %.1f s", holding.pose.label, holding.title,
                3.0 * (1 - trigger.holdProgress(now)))
            p.confidence < trigger.minConfidence ->
                "The model is not sure (below ${(trigger.minConfidence * 100).toInt()}%), so nothing happens"
            else -> "${p.best.label} has no action"
        }
    }

    // ------------------------------------------------------------------ act

    private fun switchOn(action: PoseAction) {
        when (action) {
            PoseAction.DO_NOT_DISTURB -> setDoNotDisturb(true)
            PoseAction.FLASHLIGHT -> setFlashlight(true)
            PoseAction.TIMER -> timerStartedAt = System.currentTimeMillis()
            PoseAction.DESK_CLOCK -> {}
        }
        feedback.beep(ToneGenerator.TONE_PROP_BEEP2, 250)
        feedback.vibrate(150)
        addLog("${action.title} ON (${ModelNames.label(selected)} saw ${action.pose.label} for 3 s)")
        showActions()
    }

    private fun switchOff(action: PoseAction) {
        when (action) {
            PoseAction.DO_NOT_DISTURB -> setDoNotDisturb(false)
            PoseAction.FLASHLIGHT -> setFlashlight(false)
            PoseAction.TIMER -> addLog("Timer stopped at " + timerText())
            PoseAction.DESK_CLOCK -> {}
        }
        feedback.beep(ToneGenerator.TONE_PROP_BEEP, 120)
        addLog("${action.title} OFF")
        showActions()
    }

    private fun switchEverythingOff(reason: String) {
        val was = trigger.reset() ?: return
        switchOff(was)
        addLog("(everything off: $reason)")
    }

    /** Do Not Disturb needs a one-time permission ("Do Not Disturb access" in Settings). */
    private fun hasDndAccess(): Boolean =
        getSystemService(NotificationManager::class.java)?.isNotificationPolicyAccessGranted == true

    private fun setDoNotDisturb(on: Boolean) {
        val nm = getSystemService(NotificationManager::class.java) ?: return
        if (!nm.isNotificationPolicyAccessGranted) { addLog("Do Not Disturb: no access, so it is only shown here"); return }
        if (on) {
            dndBefore = nm.currentInterruptionFilter
            nm.setInterruptionFilter(NotificationManager.INTERRUPTION_FILTER_PRIORITY)
        } else {
            nm.setInterruptionFilter(dndBefore ?: NotificationManager.INTERRUPTION_FILTER_ALL)   // back to how it was
            dndBefore = null
        }
    }

    /** The back camera's flash used as a torch (needs no permission). */
    private fun setFlashlight(on: Boolean) {
        val cm = getSystemService(CameraManager::class.java) ?: return
        try {
            val id = cm.cameraIdList.firstOrNull { cid ->
                val c = cm.getCameraCharacteristics(cid)
                c.get(CameraCharacteristics.FLASH_INFO_AVAILABLE) == true &&
                    c.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            } ?: return addLog("This phone has no flashlight")
            cm.setTorchMode(id, on)
        } catch (e: CameraAccessException) {
            addLog("Flashlight not available (${e.message})")
        } catch (e: IllegalArgumentException) {
            addLog("Flashlight not available (${e.message})")
        }
    }

    // ------------------------------------------------------------------ display

    /** The big card: which action is on, with the clock or the timer. */
    private fun showActive() {
        val a = trigger.active
        activeCard.background = if (a == null) idleBackground else activeBackground
        if (a == null) {
            activeTitle.text = "No action"
            activeTitle.setTextColor(getColor(R.color.muted))
            activeBig.text = ""
            activeHint.text = "Try: screen down, screen up, upside down, or left edge up"
            return
        }
        activeTitle.text = "${a.title} ON"
        activeTitle.setTextColor(getColor(R.color.ok))
        activeBig.text = when (a) {
            PoseAction.DESK_CLOCK -> SimpleDateFormat("HH:mm", Locale.US).format(Date())
            PoseAction.TIMER -> timerText()
            PoseAction.FLASHLIGHT -> "💡"
            PoseAction.DO_NOT_DISTURB -> "🔕"
        }
        activeHint.text = a.effect.replaceFirstChar { it.uppercase() } + " · " + a.pose.label + " (client ${a.client}'s pose)"
    }

    private fun timerText(): String {
        val s = (System.currentTimeMillis() - timerStartedAt) / 1000
        return String.format(Locale.US, "%d:%02d", s / 60, s % 60)
    }

    /** The list of the four actions, with the one that is on marked. */
    private fun showActions() {
        for ((a, view) in actionRows) {
            val on = trigger.active == a
            view.text = (if (on) "● " else "○ ") + "${a.pose.label} → ${a.title}   ·  client ${a.client}'s pose"
            view.setTextColor(getColor(if (on) R.color.ok else android.R.color.black))
            view.setTypeface(null, if (on) Typeface.BOLD else Typeface.NORMAL)
        }
    }

    private fun addLog(message: String) {
        log.addFirst(SimpleDateFormat("HH:mm:ss", Locale.US).format(Date()) + "  " + message)
        while (log.size > 8) log.removeLast()
        logText.text = log.joinToString("\n")
    }
}
