package com.posefl.core

import kotlin.math.sqrt

/**
 * Raw recording format and feature extraction. Pure Kotlin (no Android imports) so it can be
 * unit-tested on a laptop. Mirrors reference/posefl_ref/features.py exactly.
 *
 * One row per accelerometer event (~50 Hz), latest value of the other sensors held:
 *   t_ms, ax, ay, az, gx, gy, gz, wx, wy, wz, rx, ry, rz, rw
 */
object SensorCsv {
    const val HEADER = "t_ms,ax,ay,az,gx,gy,gz,wx,wy,wz,rx,ry,rz,rw"
    const val N_COLS = 14

    fun parse(text: String): List<DoubleArray> {
        val rows = ArrayList<DoubleArray>()
        for (line in text.lineSequence()) {
            val s = line.trim()
            if (s.isEmpty() || s[0].isLetter()) continue // blank line or header
            val parts = s.split(',')
            if (parts.size < N_COLS) continue
            rows.add(DoubleArray(N_COLS) { parts[it].toDouble() })
        }
        return rows
    }

    fun formatRow(row: DoubleArray): String {
        val sb = StringBuilder()
        sb.append(String.format(java.util.Locale.US, "%.1f", row[0]))
        for (i in 1 until N_COLS) {
            sb.append(',')
            sb.append(String.format(java.util.Locale.US, "%.6f", row[i]))
        }
        return sb.toString()
    }
}

/**
 * How a recording is cut into training examples: one example = one window of [windowMs] (1 s),
 * a new window starts every [stepMs] (0.5 s, so neighbours overlap by half). The first and last
 * [trimMs] of a session are dropped because the phone is still moving into / out of the pose,
 * and a window needs at least [minSamples] sensor rows.
 */
data class WindowConfig(
    val windowMs: Double = 1000.0,
    val stepMs: Double = 500.0,
    val trimMs: Double = 1000.0,
    val minSamples: Int = 25,
)

/**
 * Which sensors feed the model. A feature set is a list of blocks joined by "+", e.g. "acc+grav".
 * Every block gives 6 numbers per window: the mean of x, y, z (WHICH WAY the phone points) and the
 * standard deviation of x, y, z (HOW MUCH it shakes). Baseline "acc+grav" = 12 features.
 * The sensor study (our technical contribution) compares the four sets in [STUDY].
 */
object FeatureSets {
    const val G = 9.81
    val BLOCKS = listOf("acc", "grav", "gyro", "rot")
    const val BASELINE = "acc+grav"
    val STUDY = listOf("acc", "acc+grav", "acc+grav+gyro", "acc+grav+gyro+rot")

    fun parse(spec: String): List<String> {
        val blocks = spec.split('+').map { it.trim() }.filter { it.isNotEmpty() }
        require(blocks.isNotEmpty()) { "empty feature set" }
        for (b in blocks) require(b in BLOCKS) { "unknown feature block '$b'" }
        return blocks
    }

    fun size(spec: String): Int = 6 * parse(spec).size

    fun names(spec: String): List<String> = parse(spec).flatMap { b ->
        listOf("x", "y", "z").map { "${b}_mean_$it" } + listOf("x", "y", "z").map { "${b}_std_$it" }
    }
}

object FeatureExtractor {

    /** Device-frame "up" vector from a rotation-vector quaternion (x, y, z, w):
     *  third row of Android's SensorManager.getRotationMatrixFromVector. Heading-independent. */
    fun upFromQuaternion(x: Double, y: Double, z: Double, w: Double): DoubleArray = doubleArrayOf(
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    )

    private fun lowerBound(t: DoubleArray, value: Double): Int {
        var lo = 0
        var hi = t.size
        while (lo < hi) {
            val mid = (lo + hi) ushr 1
            if (t[mid] < value) lo = mid + 1 else hi = mid
        }
        return lo
    }

    /** Cuts one recording into windows. Returns the row indexes [lo, hi) of every window. */
    fun windowRanges(rows: List<DoubleArray>, cfg: WindowConfig = WindowConfig()): List<IntRange> {
        if (rows.isEmpty()) return emptyList()
        val t = DoubleArray(rows.size) { rows[it][0] }
        val start = t[0] + cfg.trimMs
        val endLimit = t[t.size - 1] - cfg.trimMs
        val out = ArrayList<IntRange>()
        var k = 0
        while (true) {
            val ws = start + k * cfg.stepMs
            val we = ws + cfg.windowMs
            if (we > endLimit + 1e-9) break
            val lo = lowerBound(t, ws)
            val hi = lowerBound(t, we)
            if (hi - lo >= cfg.minSamples) out.add(lo until hi)
            k++
        }
        return out
    }

    /** The x, y, z values of one sensor block in one raw row (accelerometer and gravity are scaled to g). */
    private fun blockValues(row: DoubleArray, block: String, out: DoubleArray) {
        when (block) {
            "acc" -> { out[0] = row[1] / FeatureSets.G; out[1] = row[2] / FeatureSets.G; out[2] = row[3] / FeatureSets.G }
            "grav" -> { out[0] = row[4] / FeatureSets.G; out[1] = row[5] / FeatureSets.G; out[2] = row[6] / FeatureSets.G }
            "gyro" -> { out[0] = row[7]; out[1] = row[8]; out[2] = row[9] }
            "rot" -> {
                val u = upFromQuaternion(row[10], row[11], row[12], row[13])
                out[0] = u[0]; out[1] = u[1]; out[2] = u[2]
            }
            else -> throw IllegalArgumentException(block)
        }
    }

    /**
     * Turns one window of raw rows into the feature vector the model sees:
     * for every block in [spec]: mean x, mean y, mean z, then std x, std y, std z.
     */
    fun windowFeatures(rows: List<DoubleArray>, range: IntRange, spec: String): DoubleArray {
        val blocks = FeatureSets.parse(spec)
        val feats = DoubleArray(6 * blocks.size)
        val n = range.last - range.first + 1
        val v = DoubleArray(3)
        for ((bi, block) in blocks.withIndex()) {
            val sum = DoubleArray(3)
            for (i in range) {
                blockValues(rows[i], block, v)
                for (a in 0..2) sum[a] += v[a]
            }
            val mean = DoubleArray(3) { sum[it] / n }
            val sq = DoubleArray(3)
            for (i in range) {
                blockValues(rows[i], block, v)
                for (a in 0..2) { val d = v[a] - mean[a]; sq[a] += d * d }
            }
            for (a in 0..2) {
                feats[bi * 6 + a] = mean[a]
                feats[bi * 6 + 3 + a] = sqrt(sq[a] / n)
            }
        }
        return feats
    }

    /** One whole recording -> one feature vector per window. */
    fun extract(rows: List<DoubleArray>, spec: String = FeatureSets.BASELINE, cfg: WindowConfig = WindowConfig()): List<DoubleArray> =
        windowRanges(rows, cfg).map { windowFeatures(rows, it, spec) }
}
