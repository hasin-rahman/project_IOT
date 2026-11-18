package com.posefl.app

import android.app.Activity
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.Button
import android.widget.EditText
import android.widget.RadioGroup
import android.widget.TextView
import android.widget.Toast
import com.posefl.core.FeatureSets
import com.posefl.core.Pose
import org.json.JSONObject
import java.util.concurrent.Executors

class MainActivity : Activity() {
    private lateinit var prefs: AppPrefs
    private lateinit var participant: EditText
    private lateinit var serverUrl: EditText
    private lateinit var serverPassword: EditText
    private lateinit var serverStatus: TextView
    private lateinit var trainPoses: TextView
    private val ui = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        prefs = AppPrefs(this)

        participant = findViewById(R.id.participant)
        serverUrl = findViewById(R.id.server_url)
        serverPassword = findViewById(R.id.server_password)
        serverStatus = findViewById(R.id.server_status)
        trainPoses = findViewById(R.id.train_poses)

        participant.setText(prefs.participant)
        serverUrl.setText(prefs.serverUrl)
        serverPassword.setText(prefs.serverPassword)

        val group = findViewById<RadioGroup>(R.id.client_group)
        group.check(if (prefs.clientId == "B") R.id.client_b else R.id.client_a)
        group.setOnCheckedChangeListener { _, id ->
            prefs.clientId = if (id == R.id.client_b) "B" else "A"
            showTrainPoses()
        }
        showTrainPoses()

        findViewById<Button>(R.id.test_server).setOnClickListener { savePrefs(); testServer() }
        findViewById<Button>(R.id.open_record).setOnClickListener {
            savePrefs()
            if (prefs.participant.isEmpty()) {
                Toast.makeText(this, "Enter a participant ID first", Toast.LENGTH_SHORT).show()
                participant.requestFocus()
            } else {
                startActivity(Intent(this, RecordActivity::class.java))
            }
        }
        findViewById<Button>(R.id.open_sessions).setOnClickListener {
            startActivity(Intent(this, SessionsActivity::class.java))
        }
        findViewById<Button>(R.id.open_train).setOnClickListener {
            savePrefs()
            startActivity(Intent(this, TrainActivity::class.java))
        }
        findViewById<Button>(R.id.open_predict).setOnClickListener {
            savePrefs()
            startActivity(Intent(this, PredictActivity::class.java))
        }
        findViewById<Button>(R.id.open_actions).setOnClickListener {
            startActivity(Intent(this, PoseActionsActivity::class.java))
        }

        val rec = SensorRecorder(this)
        findViewById<TextView>(R.id.sensors).text =
            "Pose FL ${BuildConfig.VERSION_NAME} · ${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE}\n" +
                "Sensors: ${rec.availableSensors()}\n" +
                "Baseline features: ${FeatureSets.BASELINE} (${FeatureSets.size(FeatureSets.BASELINE)} numbers per window)"
    }

    override fun onPause() {
        super.onPause()
        savePrefs()
    }

    override fun onDestroy() {
        super.onDestroy()
        io.shutdown()
    }

    private fun savePrefs() {
        prefs.participant = participant.text.toString()
        prefs.serverUrl = serverUrl.text.toString()
        prefs.serverPassword = serverPassword.text.toString()
        participant.setText(prefs.participant)
    }

    private fun showTrainPoses() {
        val c = prefs.clientId
        trainPoses.text = "Trains on: " + Pose.trainPosesFor(c).joinToString(", ") { it.label } +
            "\nValidation and test recordings use all six poses."
    }

    private fun testServer() {
        serverStatus.setTextColor(getColor(R.color.muted))
        serverStatus.text = "Connecting to ${prefs.serverUrl} ..."
        val url = prefs.serverUrl
        val password = prefs.serverPassword
        val body = JSONObject()
            .put("client_id", prefs.clientId)
            .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
            .put("participant", prefs.participant)
        io.execute {
            val (ok, msg) = try {
                val client = ServerClient(url, password)
                val health = client.get("/health")
                val reg = client.post("/register", body)
                val tc = reg.getJSONObject("train_config")
                true to ("Connected. Run ${health.getString("run_id")}, global model v${health.getInt("version")} of " +
                    "${health.getInt("rounds")} rounds, feature set ${tc.getString("feature_set")}." +
                    if (!reg.optBoolean("expected", true)) "\nWarning: the server does not expect client ${prefs.clientId}." else "")
            } catch (e: ServerException) {
                false to (if (e.code == 401) "The server is running but the password is wrong or missing. " +
                    "Enter the password it was started with." else "Server error ${e.code}: ${e.message}")
            } catch (e: Exception) {
                false to ("Could not reach the server: ${e.message ?: e.javaClass.simpleName}\n" +
                    "Check that the server is running and the address matches the one it printed " +
                    "(on Google Cloud it starts with https://).")
            }
            ui.post {
                serverStatus.setTextColor(getColor(if (ok) R.color.ok else R.color.bad))
                serverStatus.text = msg
            }
        }
    }
}
