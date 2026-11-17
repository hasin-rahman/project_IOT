package com.fedpose.app

import android.app.Activity
import android.content.Context
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.VibrationEffect
import android.os.Vibrator
import android.text.InputType
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.view.WindowManager
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import com.fedpose.core.*
import java.io.File
import java.util.concurrent.atomic.AtomicBoolean

/**
 * One screen that does everything a client phone needs (phase 2 of the outline):
 *   Record  -> raw sensor sessions saved ON THE PHONE   (<files>/raw/<client>/<split>/<pose>__<user>__<n>.csv)
 *   Step 1  -> local-only baseline model trained on this phone's own data
 *   Step 2  -> federated training (only model deltas + counts go to the server)
 *   Step 3  -> evaluate local_A, local_B and global on this phone's test data, report metrics only
 *   Live    -> real time pose prediction for the demo
 * Raw data leaves the phone only by USB (`adb pull`) for the team's offline experiments.
 */
class MainActivity : Activity() {
    private val ui = Handler(Looper.getMainLooper())
    private lateinit var recorder: SensorRecorder
    private lateinit var root: File

    private lateinit var serverUrl: EditText
    private lateinit var userName: EditText
    private lateinit var clientGroup: RadioGroup
    private lateinit var poseSpinner: Spinner
    private lateinit var splitSpinner: Spinner
    private lateinit var secondsEdit: EditText
    private lateinit var featureSpinner: Spinner
    private lateinit var codecSpinner: Spinner
    private lateinit var epochsEdit: EditText
    private lateinit var lrEdit: EditText
    private lateinit var seedEdit: EditText
    private lateinit var adaptiveCheck: CheckBox
    private lateinit var thresholdEdit: EditText
    private lateinit var liveModelSpinner: Spinner
    private lateinit var status: TextView
    private lateinit var bigLabel: TextView
    private lateinit var logView: TextView
    private lateinit var logScroll: ScrollView
    private lateinit var fedButton: Button
    private lateinit var liveButton: Button
    private lateinit var mainView: View
    private var blindView: View? = null
    private lateinit var blindResult: TextView
    private lateinit var blindButton: Button
    private lateinit var expectedSpinner: Spinner
    @Volatile private var lastFlClient: FlClient? = null
    private val blindBuf = SessionBuffer()
    private var placements = 0
    private var gCorrect = 0
    private var aCorrect = 0; private var aTotal = 0; private var aWrongOther = 0; private var aOtherTotal = 0
    private var bCorrect = 0; private var bTotal = 0; private var bWrongOther = 0; private var bOtherTotal = 0
    private var gTotal = 0

    private val busy = AtomicBoolean(false)
    private val stopFed = AtomicBoolean(false)
    private var live: LivePredictor? = null
    private var liveRunning = false

    private val clientId: String get() = if (clientGroup.checkedRadioButtonId == idB) "B" else "A"
    private var idA = 0
    private var idB = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        root = getExternalFilesDir(null) ?: filesDir
        recorder = SensorRecorder(this)
        mainView = buildUi()
        setContentView(mainView)
        log("Data folder: ${root.absolutePath}")
        log("Pull it with: adb pull ${root.absolutePath} ./phone_data")
        val miss = recorder.missing()
        if (miss.isNotEmpty()) log("WARNING: this phone has no sensor for: $miss (pick a feature set without them)")
    }

    override fun onDestroy() {
        stopLive(); stopFed.set(true); super.onDestroy()
    }

    // ------------------------------------------------------------------------------ UI
    private fun dp(v: Int) = (v * resources.displayMetrics.density).toInt()

    private fun label(t: String) = TextView(this).apply { text = t; setTypeface(null, Typeface.BOLD); setPadding(0, dp(12), 0, dp(2)) }
    private fun edit(hint: String, value: String, number: Boolean = false) = EditText(this).apply {
        this.hint = hint; setText(value); setSingleLine()
        if (number) inputType = InputType.TYPE_CLASS_NUMBER or InputType.TYPE_NUMBER_FLAG_DECIMAL
    }
    private fun spinner(items: List<String>, selected: Int = 0) = Spinner(this).apply {
        adapter = ArrayAdapter(this@MainActivity, android.R.layout.simple_spinner_dropdown_item, items); setSelection(selected)
    }
    private fun button(t: String, onClick: () -> Unit) = Button(this).apply { text = t; setAllCaps(false); setOnClickListener { onClick() } }

    private fun buildUi(): View {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16), dp(12), dp(16), dp(24)) }
        col.addView(TextView(this).apply { text = "FedPose client"; textSize = 22f; setTypeface(null, Typeface.BOLD) })

        col.addView(label("Server URL (same Wi-Fi as the laptop)"))
        serverUrl = edit("http://192.168.1.50:8080", "http://192.168.1.50:8080"); col.addView(serverUrl)

        col.addView(label("This phone is"))
        clientGroup = RadioGroup(this).apply { orientation = LinearLayout.HORIZONTAL }
        val a = RadioButton(this).apply { text = "Client A (poses 0-2)"; id = View.generateViewId() }
        val b = RadioButton(this).apply { text = "Client B (poses 3-5)"; id = View.generateViewId() }
        idA = a.id; idB = b.id
        clientGroup.addView(a); clientGroup.addView(b); clientGroup.check(idA)
        col.addView(clientGroup)

        col.addView(label("Your name (user id in file names)"))
        userName = edit("name", "user1"); col.addView(userName)

        // ---- record
        col.addView(label("1. Record data  (training only for this client's poses)"))
        poseSpinner = spinner(Constants.POSES); col.addView(poseSpinner)
        splitSpinner = spinner(listOf("train", "val", "test", "newuser")); col.addView(splitSpinner)
        secondsEdit = edit("seconds", "25", true); col.addView(secondsEdit)
        col.addView(button("Record session (3 s countdown, then place the phone)") { recordSession() })
        col.addView(button("Show recorded data summary") { showSummary() })

        // ---- learning settings
        col.addView(label("2. Learning settings (must match on both phones and the server)"))
        featureSpinner = spinner(Constants.FEATURE_SETS.keys.toList(), Constants.FEATURE_SETS.keys.indexOf(Constants.DEFAULT_FEATURE_SET))
        col.addView(featureSpinner)
        codecSpinner = spinner(listOf(CODEC_F32, CODEC_INT8)); col.addView(codecSpinner)
        epochsEdit = edit("local epochs", "2", true); col.addView(epochsEdit)
        lrEdit = edit("learning rate", "0.3", true); col.addView(lrEdit)
        seedEdit = edit("seed", "0", true); col.addView(seedEdit)
        adaptiveCheck = CheckBox(this).apply { text = "Adaptive participation (skip small updates)" }; col.addView(adaptiveCheck)
        thresholdEdit = edit("skip threshold (L2 norm of update)", "0.5", true); col.addView(thresholdEdit)

        // ---- pipeline
        col.addView(label("3. Train and compare"))
        col.addView(button("Step 1: train local-only baseline") { trainLocal() })
        fedButton = button("Step 2: join federated training") { toggleFederated() }; col.addView(fedButton)
        col.addView(button("Step 3: evaluate local A / local B / global") { evaluateAll() })

        // ---- live
        col.addView(label("4. Live prediction (demo)"))
        liveModelSpinner = spinner(listOf("global", "local")); col.addView(liveModelSpinner)
        liveButton = button("Start live prediction") { toggleLive() }; col.addView(liveButton)
        bigLabel = TextView(this).apply { text = "-"; textSize = 34f; gravity = Gravity.CENTER; setTypeface(null, Typeface.BOLD); setPadding(0, dp(16), 0, dp(16)) }
        col.addView(bigLabel)

        // ---- demo day
        col.addView(label("5. Demo day"))
        col.addView(button("Load all three models onto this phone (do this before your slot)") { loadAllModels() })
        col.addView(button("Open blind-test screen (judges use this)") { openBlindScreen() })
        col.addView(button("Show what this phone last uploaded") { showLastUpload() })

        status = TextView(this).apply { setPadding(0, dp(8), 0, 0) }; col.addView(status)
        logView = TextView(this).apply { textSize = 11f; typeface = Typeface.MONOSPACE }
        logScroll = ScrollView(this).apply {
            addView(logView); layoutParams = LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(260))
        }
        col.addView(label("Log")); col.addView(logScroll)
        return ScrollView(this).apply { addView(col) }
    }

    private fun log(msg: String) = ui.post {
        logView.append(msg + "\n")
        if (logView.length() > 20000) logView.text = logView.text.takeLast(15000)
        logScroll.post { logScroll.fullScroll(View.FOCUS_DOWN) }
    }

    private fun setStatus(s: String) = ui.post { status.text = s }

    // ------------------------------------------------------------------------ settings
    private val featureSet: String get() = featureSpinner.selectedItem as String
    private fun trainCfg() = TrainConfig(
        epochs = epochsEdit.text.toString().toIntOrNull() ?: 2,
        lr = lrEdit.text.toString().toDoubleOrNull() ?: 0.3,
    )
    private fun clientCfg() = ClientConfig(
        clientId = clientId, serverUrl = serverUrl.text.toString().trim(), featureSet = featureSet, train = trainCfg(),
        codec = codecSpinner.selectedItem as String,
        policy = ParticipationPolicy(adaptiveCheck.isChecked, thresholdEdit.text.toString().toDoubleOrNull() ?: 0.5),
        baseSeed = seedEdit.text.toString().toLongOrNull() ?: 0,
    )

    private fun modelFile(name: String) = File(root, "models/$name.w")
    private fun saveModel(name: String, w: DoubleArray) { modelFile(name).parentFile?.mkdirs(); modelFile(name).writeText(Wire.f32ToB64(w)) }
    private fun loadModel(name: String): DoubleArray? = modelFile(name).takeIf { it.exists() }?.let { Wire.b64ToF32(it.readText().trim()) }

    /** Runs [block] on a worker thread. Returns false (and does nothing) if another task is running. */
    private fun background(name: String, block: () -> Unit): Boolean {
        if (!busy.compareAndSet(false, true)) { log("Busy, wait for the current task to finish."); return false }
        Thread {
            try { block() } catch (e: Throwable) { log("$name failed: ${e.javaClass.simpleName}: ${e.message}") }
            finally { busy.set(false); setStatus("idle") }
        }.start()
        return true
    }

    // ---------------------------------------------------------------------- recording
    private fun recordSession() {
        val poseIdx = poseSpinner.selectedItemPosition
        val split = splitSpinner.selectedItem as String
        val seconds = (secondsEdit.text.toString().toDoubleOrNull() ?: 25.0).coerceIn(5.0, 120.0)
        if (split == "train" && poseIdx !in Constants.CLIENT_POSES.getValue(clientId)) {
            log("Client $clientId may only record TRAIN data for poses ${Constants.CLIENT_POSES.getValue(clientId).map { Constants.POSES[it] }}. " +
                "Use val/test/newuser for the other poses."); return
        }
        if (liveRunning) { log("Stop live prediction first."); return }
        val missing = recorder.missing().filter { it in Constants.FEATURE_SETS.getValue("acc_grav_gyro_rot") }
        if (missing.isNotEmpty()) log("Note: missing sensors $missing; those columns will be absent from the file.")
        background("record") {
            val pose = Constants.POSES[poseIdx]
            for (i in 3 downTo 1) { setStatus("Place the phone as '$pose' ... $i"); Thread.sleep(1000) }
            val sb = StringBuilder("t_ns,sensor,x,y,z\n")
            var n = 0L
            recorder.sink = { s, t, x, y, z -> synchronized(sb) { sb.append(t).append(',').append(s).append(',').append(x).append(',').append(y).append(',').append(z).append('\n'); n++ } }
            recorder.start()
            buzz()
            val end = System.currentTimeMillis() + (seconds * 1000).toLong()
            while (System.currentTimeMillis() < end) {
                setStatus("Recording '$pose' (${split}) ... ${(end - System.currentTimeMillis()) / 1000 + 1}s left, $n samples")
                Thread.sleep(250)
            }
            recorder.stop(); recorder.sink = null
            buzz()
            val dir = File(root, "raw/$clientId/$split").also { it.mkdirs() }
            val user = userName.text.toString().trim().replace(Regex("[^A-Za-z0-9]"), "").ifEmpty { "user" }
            val k = (dir.listFiles { f -> f.name.startsWith("${pose}__${user}__") }?.size ?: 0)
            val f = File(dir, "${pose}__${user}__${"%02d".format(k)}.csv")
            synchronized(sb) { f.writeText(sb.toString()) }
            log("Saved ${f.name}: $n samples ($split, client $clientId). Raw data stays on this phone.")
        }
    }

    private fun buzz() {
        val v = getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator ?: return
        v.vibrate(VibrationEffect.createOneShot(150, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    private fun showSummary() = background("summary") {
        val sb = StringBuilder("Recorded sessions for client $clientId:\n")
        for (split in listOf("train", "val", "test", "newuser")) {
            val files = Dataset.sessionFiles(root, clientId, split)
            val perPose = files.groupingBy { Constants.POSES[it.second] }.eachCount()
            sb.append("  $split: ${files.size} sessions ").append(perPose.entries.joinToString { "${it.key}=${it.value}" }).append('\n')
        }
        val d = Dataset.load(root, clientId, "train", featureSet)
        sb.append("  -> ${d.size} training windows with feature set $featureSet (${Constants.nFeatures(featureSet)} features)")
        log(sb.toString())
    }

    // ------------------------------------------------------------------- local + FL
    private fun trainLocal() = background("local training") {
        setStatus("Training local-only baseline ...")
        val cfg = clientCfg()
        val data = Dataset.load(root, clientId, "train", featureSet) { log(it) }
        if (data.size == 0) { log("No training data yet. Record sessions first."); return@background }
        val client = FlClient(cfg, data, log = { log(it) })
        val rounds = serverRounds(cfg) ?: 30
        val w = client.trainLocalOnly(rounds)
        saveModel("local", w)
        saveModel("local_$clientId", w)  // blind test loads local_A, local_B and global by name
        val (loss, acc) = client.model.lossAcc(w, data)
        log("Local-only model trained: ${rounds * cfg.train.epochs} epochs on ${data.size} windows (loss %.3f, train acc %.3f). Saved.".format(loss, acc))
        try { client.uploadLocalModel(w); log("Local model weights uploaded so the other phone can score it (weights only).") }
        catch (e: Exception) { log("Upload of local model skipped (${e.message}); retry after the server is up.") }
    }

    /** Number of rounds the server will run, so the local baseline gets the same total epochs. */
    private fun serverRounds(cfg: ClientConfig): Int? = try {
        FlHttp(cfg.serverUrl, 2000, 4000).post("/register", mapOf("client_id" to cfg.clientId)).body["max_rounds"]?.let { (it as Number).toInt() }
    } catch (e: Exception) { log("Server not reachable for round count (${e.message}); assuming 30 rounds."); null }

    @Volatile private var fedRunning = false

    private fun toggleFederated() {
        if (fedRunning) { stopFed.set(true); log("Stopping federated training after the current step ..."); return }
        stopFed.set(false)
        val started = background("federated training") {
            fedRunning = true
            ui.post { fedButton.text = "Stop federated training" }
            try {
                val cfg = clientCfg()
                val data = Dataset.load(root, clientId, "train", featureSet) { log(it) }
                if (data.size == 0) { log("No training data yet. Record sessions first."); return@background }
                log("Client $clientId: ${data.size} local windows, classes ${data.y.toSet().sorted()}. Raw data never leaves this phone.")
                val client = FlClient(cfg, data, log = { log(it) }, shouldStop = { stopFed.get() })
                lastFlClient = client
                client.register()
                setStatus("Federated training running (waiting for the server to start rounds) ...")
                val sum = client.runFederated()
                log("Federation finished: $sum")
                client.fetchGlobal()?.let { saveModel("global", it); log("Global model saved on this phone.") }
            } finally {
                fedRunning = false
                stopFed.set(false)
                ui.post { fedButton.text = "Step 2: join federated training" }
            }
        }
        if (!started) return
    }

    private fun evaluateAll() = background("evaluation") {
        setStatus("Evaluating all three models on this phone's test data ...")
        val local = loadModel("local") ?: run { log("Train the local-only baseline first (Step 1)."); return@background }
        val cfg = clientCfg()
        val tests = linkedMapOf(
            "own_test" to Dataset.load(root, clientId, "test", featureSet),
            "newuser" to Dataset.load(root, clientId, "newuser", featureSet),
        ).filterValues { it.size > 0 }
        if (tests.isEmpty()) { log("No test/newuser sessions recorded for this phone."); return@background }
        val client = FlClient(cfg, LocalData(emptyList(), IntArray(0)), log = { log(it) })
        val res = client.evaluateAllAndReport(tests, local)
        res["global"]?.get("own_test")?.let { ev ->
            log("Per-pose accuracy of the global model: " + Constants.POSES.indices.joinToString { "${Constants.POSES[it]}=%.2f".format(ev.perClassRecall[it]) })
        }
        log("Only metrics were sent to the server (no sensor data, no features).")
    }

    // -------------------------------------------------------------------- demo day
    private val modelNames = listOf("global", "local_A", "local_B")
    private fun pretty(n: String) = when (n) { "global" -> "Global"; "local_A" -> "Local A"; else -> "Local B" }

    /** Puts global, local_A and local_B on this phone so the blind test needs no network. */
    private fun loadAllModels() = background("load models") {
        setStatus("Loading all three models ...")
        val cfg = clientCfg()
        val client = FlClient(cfg, LocalData(emptyList(), IntArray(0)), log = { log(it) })
        val other = if (clientId == "A") "B" else "A"
        loadModel("local")?.let { saveModel("local_$clientId", it) }   // this phone's own independent baseline
        client.fetchLocalModel(other)?.let { saveModel("local_$other", it); log("local_$other received from the server (weights only)") }
        client.fetchGlobal()?.let { saveModel("global", it); log("global model received") }
        val have = modelNames.associateWith { modelFile(it).exists() }
        log("Models on this phone: " + have.entries.joinToString { "${it.key}=${if (it.value) "yes" else "MISSING"}" })
        if (have.values.any { !it }) log("Missing ones: train the local baseline (Step 1) on BOTH phones and finish federation (Step 2), then try again.")
    }

    private fun showLastUpload() {
        val c = lastFlClient
        log("Last upload from this phone (what the server received, payload shortened):\n" + (c?.lastUploadSummary ?: "(nothing uploaded yet; run Step 2)"))
        log("Server view of all uploads: open  " + serverUrl.text.toString().trim().trimEnd('/') + "/audit.html  in a browser.")
    }

    private fun openBlindScreen() {
        if (liveRunning) stopLive()
        val v = blindView ?: buildBlindUi().also { blindView = it }
        setContentView(v)
    }

    private fun closeBlindScreen() { setContentView(mainView) }

    private fun buildBlindUi(): View {
        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(dp(16), dp(16), dp(16), dp(16)) }
        col.addView(TextView(this).apply { text = "Blind test"; textSize = 24f; setTypeface(null, Typeface.BOLD) })
        col.addView(TextView(this).apply { text = "Press START, place the phone during the 3 s countdown, keep it still for the 2 s recording." })
        blindResult = TextView(this).apply {
            text = "Ready"; textSize = 24f; typeface = Typeface.MONOSPACE; setPadding(0, dp(20), 0, dp(20))
        }
        col.addView(blindResult, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f))
        blindButton = button("START PLACEMENT") { runPlacement() }.apply { textSize = 26f }
        col.addView(blindButton, LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, dp(110)))
        col.addView(label("Rehearsal only: the true pose (judges leave this on '(judges)')"))
        expectedSpinner = spinner(listOf("(judges)") + Constants.POSES); col.addView(expectedSpinner)
        col.addView(button("Reset tally") { resetTally(); blindResult.text = "Ready" })
        col.addView(button("Back") { closeBlindScreen() })
        return col
    }

    private fun resetTally() {
        placements = 0; gCorrect = 0; gTotal = 0
        aCorrect = 0; aTotal = 0; aWrongOther = 0; aOtherTotal = 0
        bCorrect = 0; bTotal = 0; bWrongOther = 0; bOtherTotal = 0
    }

    private fun setBlind(t: String) = ui.post { blindResult.text = t }

    /** One judge placement: 3 s countdown, 2 s recording, three predictions from models stored on the phone. */
    private fun runPlacement() {
        val names = modelNames
        val models = names.associateWith { loadModel(it) }
        val missing = models.filterValues { it == null }.keys
        if (missing.isNotEmpty()) { setBlind("Missing models: $missing\nLoad them on the main screen first."); return }
        val need = Constants.nParams(Constants.nFeatures(featureSet))
        if (models.values.any { it!!.size != need }) { setBlind("Models do not match feature set $featureSet.\nRetrain with the same feature set."); return }
        val truth = expectedSpinner.selectedItemPosition - 1   // -1 = judges
        val started = background("blind test") {
            val n = ++placements
            val sm = SoftmaxModel(Constants.nFeatures(featureSet))
            blindBuf.clear()
            recorder.sink = { s, t, x, y, z -> blindBuf.push(s, t, x, y, z) }
            recorder.start()
            for (i in 3 downTo 1) { setBlind("Placement #$n\n\nplace the phone\n\n        $i"); Thread.sleep(1000) }
            val mark = blindBuf.mark()
            buzz()
            setBlind("Placement #$n\n\nRECORDING ...")
            Thread.sleep(2000)
            recorder.stop(); recorder.sink = null
            val session = blindBuf.sessionSince(mark)
            val t0 = BlindTest.clipStart(session, featureSet)
            var wins = if (t0 == null) emptyList() else BlindTest.clipWindows(session, featureSet, t0)
            if (wins.isEmpty() && t0 != null) wins = BlindTest.clipWindows(session, featureSet, t0, doubleArrayOf(0.0, 0.25, 0.5))
            if (wins.isEmpty()) { setBlind("Placement #$n\nNo sensor data.\nTry again."); placements--; return@background }
            val preds = names.associateWith { nm -> BlindTest.clipProba(sm, models.getValue(nm)!!, wins)!! }
            val labels = preds.mapValues { BlindTest.argmax(it.value) }
            val sb = StringBuilder("Placement #$n\n\n")
            for (nm in names) {
                val l = labels.getValue(nm)
                val mark2 = if (truth < 0) "" else if (l == truth) "  OK" else "  X"
                sb.append("%-8s %s %d%%%s\n".format(pretty(nm) + ":", Constants.POSES[l], Math.round(preds.getValue(nm)[l] * 100), mark2))
            }
            if (truth >= 0) {
                gTotal++; if (labels.getValue("global") == truth) gCorrect++
                if (truth in Constants.CLIENT_POSES.getValue("A")) { aTotal++; if (labels.getValue("local_A") == truth) aCorrect++ }
                else { aOtherTotal++; if (labels.getValue("local_A") != truth) aWrongOther++ }
                if (truth in Constants.CLIENT_POSES.getValue("B")) { bTotal++; if (labels.getValue("local_B") == truth) bCorrect++ }
                else { bOtherTotal++; if (labels.getValue("local_B") != truth) bWrongOther++ }
                val ga = gCorrect.toDouble() / gTotal
                val la = if (aTotal > 0) aCorrect.toDouble() / aTotal else 0.0
                val lb = if (bTotal > 0) bCorrect.toDouble() / bTotal else 0.0
                sb.append("\nTrue pose: ${Constants.POSES[truth]}\nGlobal $gCorrect/$gTotal  A $aCorrect/$aTotal  B $bCorrect/$bTotal")
                sb.append("\nRubric Part A so far: %.1f / 40".format(30 * ga + 5 * la + 5 * lb))
                sb.append("\nLocals wrong on other poses: A $aWrongOther/$aOtherTotal, B $bWrongOther/$bOtherTotal")
            }
            setBlind(sb.toString())
            log("#$n " + names.joinToString("  ") { "${pretty(it)}=${Constants.POSES[labels.getValue(it)]}" } + if (truth >= 0) "  (true: ${Constants.POSES[truth]})" else "")
            try {
                File(root, "blind_log.csv").appendText("$n,${if (truth < 0) "" else Constants.POSES[truth]}," +
                    names.joinToString(",") { Constants.POSES[labels.getValue(it)] } + "\n")
            } catch (_: Exception) {}
        }
        if (!started) setBlind("Busy, try again in a moment.")
    }

    // -------------------------------------------------------------------------- live
    private fun toggleLive() { if (liveRunning) stopLive() else startLive() }

    private fun startLive() {
        if (busy.get()) { log("Busy, wait for the current task to finish."); return }
        val which = liveModelSpinner.selectedItem as String
        val w = loadModel(which) ?: run { log("No '$which' model on this phone yet."); return }
        val lp = LivePredictor(featureSet, w).also { live = it }
        recorder.sink = { s, t, x, y, z -> lp.push(s, t, x, y, z) }
        recorder.start()
        liveRunning = true
        liveButton.text = "Stop live prediction"
        log("Live prediction with the $which model ($featureSet).")
        val tick = object : Runnable {
            override fun run() {
                if (!liveRunning) return
                val p = live?.predict()
                bigLabel.text = if (p == null) "..." else "${Constants.POSES[p.smoothedLabel]}\n%.0f%%".format(p.smoothedProbs[p.smoothedLabel] * 100)
                ui.postDelayed(this, 500)
            }
        }
        ui.post(tick)
    }

    private fun stopLive() {
        if (!liveRunning) return
        liveRunning = false
        recorder.stop(); recorder.sink = null
        liveButton.text = "Start live prediction"
        bigLabel.text = "-"
    }
}
