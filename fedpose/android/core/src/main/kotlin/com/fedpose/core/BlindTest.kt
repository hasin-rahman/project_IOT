package com.fedpose.core

/** Collects raw sensor events in memory (ON THE PHONE) so a short clip can be cut out after the fact. */
class SessionBuffer {
    private val t = HashMap<String, ArrayList<Long>>()
    private val v = HashMap<String, ArrayList<DoubleArray>>()

    @Synchronized fun push(sensor: String, tNs: Long, x: Double, y: Double, z: Double) {
        t.getOrPut(sensor) { ArrayList() }.add(tNs)
        v.getOrPut(sensor) { ArrayList() }.add(doubleArrayOf(x, y, z))
    }

    /** Remember how many samples each sensor has now, e.g. when the countdown ends. */
    @Synchronized fun mark(): Map<String, Int> = t.mapValues { it.value.size }

    /** Everything recorded after [mark], as a Session (sorted by time). */
    @Synchronized fun sessionSince(mark: Map<String, Int>): Session {
        val out = HashMap<String, SensorSeries>()
        for ((s, tl) in t) {
            val from = mark[s] ?: 0
            val idx = (from until tl.size).sortedBy { tl[it] }
            if (idx.isEmpty()) continue
            val vs = v.getValue(s)
            out[s] = SensorSeries(
                LongArray(idx.size) { tl[idx[it]] },
                DoubleArray(idx.size) { vs[idx[it]][0] },
                DoubleArray(idx.size) { vs[idx[it]][1] },
                DoubleArray(idx.size) { vs[idx[it]][2] },
            )
        }
        return out
    }

    @Synchronized fun clear() { t.clear(); v.clear() }
}

/**
 * Demo day blind test: one pose label from a 2 s recording. Twin of python/fedpose/blind.py.
 * The clip is cut into 1 s windows (same features as training, no trimming) and the softmax
 * probabilities are averaged over the windows.
 */
object BlindTest {
    const val CLIP_S = 2.0

    /** Window start offsets (s) from the start of the recording. Default skips the first 0.5 s (hand settling). */
    val DEFAULT_STARTS = doubleArrayOf(0.5, 1.0)

    /** The clip starts when every needed sensor has produced data (same rule as Python). */
    fun clipStart(session: Session, featureSet: String): Long? {
        val sensors = Constants.FEATURE_SETS.getValue(featureSet)
        if (sensors.any { it !in session }) return null
        return sensors.maxOf { session.getValue(it).t.first() }
    }

    fun clipWindows(session: Session, featureSet: String, clipStartNs: Long,
                    starts: DoubleArray = DEFAULT_STARTS): List<DoubleArray> {
        val sensors = Constants.FEATURE_SETS.getValue(featureSet)
        val win = Math.round(Constants.WINDOW_S * 1e9)
        return starts.toList().mapNotNull { Features.windowVector(session, sensors, clipStartNs + Math.round(it * 1e9), win) }
    }

    /** Mean softmax probabilities over the clip's windows, or null if the clip had no usable window. */
    fun clipProba(model: SoftmaxModel, w: DoubleArray, windows: List<DoubleArray>): DoubleArray? {
        if (windows.isEmpty()) return null
        val acc = DoubleArray(Constants.N_CLASSES)
        for (x in windows) { val p = model.proba(w, x); for (k in acc.indices) acc[k] += p[k] }
        for (k in acc.indices) acc[k] /= windows.size
        return acc
    }

    fun argmax(p: DoubleArray): Int { var b = 0; for (k in 1 until p.size) if (p[k] > p[b]) b = k; return b }
}
