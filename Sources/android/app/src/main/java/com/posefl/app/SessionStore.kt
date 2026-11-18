package com.posefl.app

import android.content.Context
import com.posefl.core.LabeledSession
import com.posefl.core.Pose
import com.posefl.core.SensorCsv
import org.json.JSONObject
import java.io.File

/** Metadata saved next to each raw recording. */
data class SessionMeta(
    val id: String,
    val client: String,
    val participant: String,
    val pose: Pose,
    val split: String,        // "train", "val" or "test"
    val holding: String,      // "hand", "surface" or "propped"
    val startedAt: Long,      // epoch ms
    val durationMs: Long,
    val samples: Int,
    val windows: Int,         // usable 1-second windows after trimming
    val labelMatch: Double,   // fraction of samples whose gravity agrees with the label
    val device: String,
    val sensors: String,
) {
    fun toJson(): JSONObject = JSONObject()
        .put("id", id).put("client", client).put("participant", participant)
        .put("pose", pose.key).put("pose_index", pose.index).put("split", split).put("holding", holding)
        .put("started_at", startedAt).put("duration_ms", durationMs).put("samples", samples)
        .put("windows", windows).put("label_match", labelMatch).put("device", device).put("sensors", sensors)

    companion object {
        val SPLITS = listOf("train", "val", "test")

        fun fromJson(o: JSONObject) = SessionMeta(
            id = o.getString("id"), client = o.getString("client"), participant = o.optString("participant"),
            pose = Pose.fromKey(o.getString("pose")), split = o.getString("split"),
            holding = o.optString("holding", "hand"), startedAt = o.getLong("started_at"),
            durationMs = o.getLong("duration_ms"), samples = o.getInt("samples"), windows = o.getInt("windows"),
            labelMatch = o.optDouble("label_match", 1.0), device = o.optString("device"),
            sensors = o.optString("sensors"),
        )
    }
}

/**
 * Raw recordings live in the app's private storage (filesDir/sessions). Other apps cannot read
 * them and they are never uploaded: the FL privacy rule says sensor data stays on the phone.
 */
class SessionStore(private val context: Context) {
    private val dir = File(context.filesDir, "sessions").apply { mkdirs() }

    fun save(meta: SessionMeta, rows: List<DoubleArray>) {
        File(dir, "${meta.id}.csv").bufferedWriter().use { w ->
            w.write(SensorCsv.HEADER)
            w.newLine()
            for (r in rows) {
                w.write(SensorCsv.formatRow(r))
                w.newLine()
            }
        }
        File(dir, "${meta.id}.json").writeText(meta.toJson().toString(1))
    }

    fun list(): List<SessionMeta> =
        (dir.listFiles { f -> f.name.endsWith(".json") } ?: emptyArray())
            .mapNotNull { runCatching { SessionMeta.fromJson(JSONObject(it.readText())) }.getOrNull() }
            .sortedByDescending { it.startedAt }

    fun loadRows(id: String): List<DoubleArray> = SensorCsv.parse(File(dir, "$id.csv").readText())

    /** All sessions recorded on this phone under [clientId], with their raw rows (stays on the phone). */
    fun loadLabeled(clientId: String): List<LabeledSession> = list().filter { it.client == clientId }.mapNotNull { m ->
        runCatching { LabeledSession(m.id, m.pose, m.split, m.participant, loadRows(m.id)) }.getOrNull()
    }

    fun delete(id: String) {
        File(dir, "$id.csv").delete()
        File(dir, "$id.json").delete()
    }
}
