package com.posefl.app

import android.app.Activity
import android.app.AlertDialog
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.widget.ArrayAdapter
import android.widget.ListView
import android.widget.TextView
import com.posefl.core.Pose
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.Executors

/** Lists recorded sessions and summarises how many usable windows exist per pose and split. */
class SessionsActivity : Activity() {
    private lateinit var store: SessionStore
    private lateinit var prefs: AppPrefs
    private lateinit var summary: TextView
    private lateinit var list: ListView
    private var sessions: List<SessionMeta> = emptyList()
    private val ui = Handler(Looper.getMainLooper())
    private val io = Executors.newSingleThreadExecutor()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_sessions)
        store = SessionStore(this)
        prefs = AppPrefs(this)
        summary = findViewById(R.id.summary)
        list = findViewById(R.id.list)

        list.setOnItemClickListener { _, _, pos, _ ->
            val s = sessions[pos]
            AlertDialog.Builder(this).setTitle(s.id).setMessage(s.toJson().toString(2))
                .setPositiveButton("Close", null).show()
        }
        list.setOnItemLongClickListener { _, _, pos, _ ->
            val s = sessions[pos]
            AlertDialog.Builder(this)
                .setTitle("Delete this session?")
                .setMessage("${s.pose.label} · ${s.split} · ${s.windows} windows\n${s.id}\n\nThis cannot be undone.")
                .setPositiveButton("Delete") { _, _ -> io.execute { store.delete(s.id); ui.post { reload() } } }
                .setNegativeButton("Keep", null)
                .show()
            true
        }
    }

    override fun onResume() {
        super.onResume()
        reload()
    }

    override fun onDestroy() {
        super.onDestroy()
        io.shutdown()
    }

    private fun reload() {
        io.execute {
            val all = store.list()
            val text = buildSummary(all)
            val fmt = SimpleDateFormat("MMM d HH:mm", Locale.US)
            val lines = all.map {
                val flag = if (it.labelMatch < 0.8) "  ⚠ label check ${(it.labelMatch * 100).toInt()}%" else ""
                "${it.client} · ${it.pose.label} · ${it.split} · ${it.holding}\n" +
                    "${it.participant} · ${fmt.format(Date(it.startedAt))} · ${it.windows} windows$flag"
            }
            ui.post {
                sessions = all
                summary.text = text
                list.adapter = ArrayAdapter(this, android.R.layout.simple_list_item_1, lines)
            }
        }
    }

    private fun buildSummary(all: List<SessionMeta>): String {
        val client = prefs.clientId
        val mine = all.filter { it.client == client }
        val trainPoses = Pose.trainPosesFor(client).toSet()
        val sb = StringBuilder()
        sb.append("Client $client - usable windows (sessions)\n")
        sb.append(String.format(Locale.US, "%-17s %9s %9s %9s\n", "pose", "train", "val", "test"))
        for (p in Pose.entries) {
            val cells = SessionMeta.SPLITS.map { split ->
                val ss = mine.filter { it.pose == p && it.split == split }
                if (split == "train" && p !in trainPoses) "-" else "${ss.sumOf { it.windows }} (${ss.size})"
            }
            sb.append(String.format(Locale.US, "%-17s %9s %9s %9s\n", p.label, cells[0], cells[1], cells[2]))
        }
        val people = mine.map { it.participant }.distinct()
        sb.append("\n${mine.size} sessions from ${people.size} participant(s): ${people.joinToString(", ")}")
        val other = all.size - mine.size
        if (other > 0) sb.append("\n($other sessions recorded under the other client ID)")
        return sb.toString()
    }
}
