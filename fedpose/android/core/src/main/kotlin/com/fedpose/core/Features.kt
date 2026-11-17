package com.fedpose.core

import java.io.File
import kotlin.math.sqrt

/** One sensor's samples, sorted by time. */
class SensorSeries(val t: LongArray, val x: DoubleArray, val y: DoubleArray, val z: DoubleArray) {
    val size: Int get() = t.size
}

typealias Session = Map<String, SensorSeries>

/**
 * Raw session CSV -> windowed features. Kotlin twin of python/fedpose/features.py.
 *
 * Raw CSV: header `t_ns,sensor,x,y,z`, rows like `123456789000,acc,0.01,0.02,9.80`.
 * Per sensor a window contributes mean(x,y,z) then std(x,y,z) (population std) of the values
 * divided by the sensor's fixed scale; sensors are concatenated in feature-set order.
 */
object Features {
    fun parseSession(lines: Iterable<String>): Session {
        val ts = HashMap<String, ArrayList<Long>>()
        val xs = HashMap<String, ArrayList<DoubleArray>>()
        var first = true
        for (raw in lines) {
            if (first) {
                first = false
                val h = raw.split(',').map { it.trim() }
                require(h == listOf("t_ns", "sensor", "x", "y", "z")) { "unexpected header $h" }
                continue
            }
            if (raw.isBlank()) continue
            val p = raw.split(',')
            val s = p[1].trim()
            if (s !in Constants.SENSORS) continue
            ts.getOrPut(s) { ArrayList() }.add(p[0].trim().toLong())
            xs.getOrPut(s) { ArrayList() }.add(doubleArrayOf(p[2].toDouble(), p[3].toDouble(), p[4].toDouble()))
        }
        val out = HashMap<String, SensorSeries>()
        for ((s, tl) in ts) {
            val order = tl.indices.sortedBy { tl[it] }  // stable
            val n = order.size
            val v = xs.getValue(s)
            out[s] = SensorSeries(
                LongArray(n) { tl[order[it]] },
                DoubleArray(n) { v[order[it]][0] },
                DoubleArray(n) { v[order[it]][1] },
                DoubleArray(n) { v[order[it]][2] },
            )
        }
        return out
    }

    fun loadSessionCsv(f: File): Session = f.useLines { parseSession(it.asIterable()) }

    /** First index with t[i] >= v (numpy searchsorted side="left"). */
    fun lowerBound(t: LongArray, v: Long): Int {
        var lo = 0
        var hi = t.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (t[mid] < v) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** Features of the window [startNs, startNs + winNs), or null if any sensor has too few samples. */
    fun windowVector(session: Session, sensors: List<String>, startNs: Long, winNs: Long,
                     minSamples: Int = Constants.MIN_SAMPLES): DoubleArray? {
        val out = DoubleArray(6 * sensors.size)
        var o = 0
        for (s in sensors) {
            val ser = session[s] ?: return null
            val lo = lowerBound(ser.t, startNs)
            val hi = lowerBound(ser.t, startNs + winNs)
            val n = hi - lo
            if (n < minSamples) return null
            val sc = Constants.SENSOR_SCALE.getValue(s)
            val axes = arrayOf(ser.x, ser.y, ser.z)
            for (a in 0 until 3) {
                var sum = 0.0
                for (i in lo until hi) sum += axes[a][i] / sc
                val mean = sum / n
                var sq = 0.0
                for (i in lo until hi) { val d = axes[a][i] / sc - mean; sq += d * d }
                out[o + a] = mean
                out[o + 3 + a] = sqrt(sq / n)
            }
            o += 6
        }
        return out
    }

    fun windowFeatures(
        session: Session,
        featureSet: String = Constants.DEFAULT_FEATURE_SET,
        windowS: Double = Constants.WINDOW_S,
        strideS: Double = Constants.STRIDE_S,
        trimS: Double = Constants.TRIM_S,
        minSamples: Int = Constants.MIN_SAMPLES,
    ): List<DoubleArray> {
        val sensors = Constants.FEATURE_SETS.getValue(featureSet)
        if (sensors.any { it !in session }) return emptyList()
        val win = Math.round(windowS * 1e9)
        val stride = Math.round(strideS * 1e9)
        val trim = Math.round(trimS * 1e9)
        val tLo = sensors.maxOf { session.getValue(it).t.first() } + trim
        val tHi = sensors.minOf { session.getValue(it).t.last() } - trim
        val out = ArrayList<DoubleArray>()
        var start = tLo
        while (start + win <= tHi) {
            windowVector(session, sensors, start, win, minSamples)?.let { out.add(it) }
            start += stride
        }
        return out
    }
}
