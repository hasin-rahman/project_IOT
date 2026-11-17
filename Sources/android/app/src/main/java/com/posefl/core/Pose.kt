package com.posefl.core

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The six phone poses. Android axes: +x = right edge, +y = top edge, +z = out of the screen.
 * A resting gravity sensor reads +9.81 along whichever axis points UP, so each pose has a
 * dominant gravity axis and sign.
 */
enum class Pose(
    val index: Int,
    val key: String,
    val label: String,
    val description: String,
    val axis: Int,
    val sign: Int,
) {
    SCREEN_UP(0, "screen_up", "Screen Up", "Lying flat, screen facing up", 2, +1),
    SCREEN_DOWN(1, "screen_down", "Screen Down", "Lying flat, screen facing down", 2, -1),
    PORTRAIT_UP(2, "portrait_up", "Portrait Up", "Vertical, top edge pointing up", 1, +1),
    PORTRAIT_DOWN(3, "portrait_down", "Portrait Down", "Vertical, top edge pointing down", 1, -1),
    LANDSCAPE_LEFT(4, "landscape_left", "Landscape Left", "Vertical, left edge pointing up", 0, -1),
    LANDSCAPE_RIGHT(5, "landscape_right", "Landscape Right", "Vertical, right edge pointing up", 0, +1);

    companion object {
        const val COUNT = 6

        fun fromIndex(i: Int): Pose = entries.first { it.index == i }
        fun fromKey(k: String): Pose = entries.first { it.key == k }

        /** Poses each client may use for TRAINING (non-IID split). Both collect all six for val/test. */
        fun trainPosesFor(client: String): List<Pose> = when (client) {
            "A" -> listOf(SCREEN_UP, SCREEN_DOWN, PORTRAIT_UP)
            "B" -> listOf(PORTRAIT_DOWN, LANDSCAPE_LEFT, LANDSCAPE_RIGHT)
            else -> entries.toList()
        }

        /**
         * Rule-based guess from a gravity reading. Used ONLY as a recording-time label check
         * (to catch mislabeled sessions), never as the classifier. Returns null when the phone
         * is too close to 45 degrees between two poses to tell.
         */
        fun guessFromGravity(gx: Double, gy: Double, gz: Double, minDominance: Double = 0.75): Pose? {
            val norm = sqrt(gx * gx + gy * gy + gz * gz)
            if (norm < 1e-6) return null
            val v = doubleArrayOf(gx, gy, gz)
            var axis = 0
            for (i in 1..2) if (abs(v[i]) > abs(v[axis])) axis = i
            if (abs(v[axis]) / norm < minDominance) return null
            val sign = if (v[axis] >= 0) 1 else -1
            return entries.first { it.axis == axis && it.sign == sign }
        }
    }
}
