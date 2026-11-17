package com.fedpose.core

/** One live prediction. `label` is the raw argmax, `smoothedLabel` uses the EMA over recent windows. */
class Prediction(val label: Int, val probs: DoubleArray, val smoothedLabel: Int, val smoothedProbs: DoubleArray)

/**
 * Real time pose prediction for the in-class demo. Feed every sensor event into [push]; call
 * [predict] every ~500 ms. It uses exactly the training feature code (1 s window ending at the
 * newest common timestamp), so live features match the training distribution.
 * Thread safe: [push] is called from the sensor thread, [predict] from the UI/timer thread.
 */
class LivePredictor(
    val featureSet: String,
    weights: DoubleArray,
    private val smoothing: Double = 0.5,
    private val keepNs: Long = 3_000_000_000L,
) {
    private val sensors = Constants.FEATURE_SETS.getValue(featureSet)
    private val nf = Constants.nFeatures(featureSet)
    private val model = SoftmaxModel(nf)
    @Volatile private var w: DoubleArray = weights
    private class Sample(val t: Long, val x: Double, val y: Double, val z: Double)
    private val bufs = sensors.associateWith { ArrayDeque<Sample>() }
    private var ema: DoubleArray? = null

    fun setWeights(newW: DoubleArray) { require(newW.size == model.nParams); w = newW; synchronized(this) { ema = null } }

    @Synchronized fun reset() { bufs.values.forEach { it.clear() }; ema = null }

    @Synchronized fun push(sensor: String, tNs: Long, x: Double, y: Double, z: Double) {
        val b = bufs[sensor] ?: return
        b.addLast(Sample(tNs, x, y, z))
        while (b.isNotEmpty() && tNs - b.first().t > keepNs) b.removeFirst()
    }

    @Synchronized fun predict(): Prediction? {
        if (bufs.values.any { it.isEmpty() }) return null
        val end = bufs.values.minOf { it.last().t } + 1
        val win = Math.round(Constants.WINDOW_S * 1e9)
        val session: Session = bufs.mapValues { (_, d) ->
            val l = d.toList()
            SensorSeries(LongArray(l.size) { l[it].t }, DoubleArray(l.size) { l[it].x },
                DoubleArray(l.size) { l[it].y }, DoubleArray(l.size) { l[it].z })
        }
        val x = Features.windowVector(session, sensors, end - win, win) ?: return null
        val p = model.proba(w, x)
        val prev = ema
        val s = if (prev == null) p.copyOf() else DoubleArray(p.size) { smoothing * p[it] + (1 - smoothing) * prev[it] }
        ema = s
        fun argmax(a: DoubleArray): Int { var b = 0; for (k in 1 until a.size) if (a[k] > a[b]) b = k; return b }
        return Prediction(argmax(p), p, argmax(s), s)
    }
}
