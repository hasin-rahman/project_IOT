package com.fedpose.core

/** Mirrors python/fedpose/constants.py. Change both together or parity tests will fail. */
object Constants {
    val POSES = listOf("screen_up", "screen_down", "portrait_up", "portrait_down", "landscape_left", "landscape_right")
    const val N_CLASSES = 6

    /** Non-IID split: A trains on the first three poses, B on the last three. */
    val CLIENT_POSES = mapOf("A" to listOf(0, 1, 2), "B" to listOf(3, 4, 5))
    val CLIENT_INDEX = mapOf("A" to 0, "B" to 1)

    val SENSORS = listOf("acc", "grav", "gyro", "rot")

    /** Fixed, client independent scaling (per-client standardization would break FedAvg under non-IID data). */
    val SENSOR_SCALE = mapOf("acc" to 9.81, "grav" to 9.81, "gyro" to Math.PI, "rot" to 1.0)

    val FEATURE_SETS = mapOf(
        "acc" to listOf("acc"),
        "acc_grav" to listOf("acc", "grav"),
        "acc_grav_gyro" to listOf("acc", "grav", "gyro"),
        "acc_grav_gyro_rot" to listOf("acc", "grav", "gyro", "rot"),
    )
    const val DEFAULT_FEATURE_SET = "acc_grav"

    const val WINDOW_S = 1.0
    const val STRIDE_S = 0.5
    const val TRIM_S = 1.0
    const val MIN_SAMPLES = 10

    fun nFeatures(featureSet: String): Int = 6 * FEATURE_SETS.getValue(featureSet).size
    fun nParams(nf: Int): Int = N_CLASSES * nf + N_CLASSES
}
