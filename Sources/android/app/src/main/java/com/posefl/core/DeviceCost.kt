package com.posefl.core

import java.util.Locale

/**
 * On-device cost: is federated learning practical on a battery-powered IoT device?
 *
 * During a run, [CostTracker] notes how long each step took on this phone, how many bytes went over
 * the network and how much battery was used. At the end of the run the phone shows the numbers and
 * sends them to the server (POST /device_cost), so the dashboard can compare plain, secure and
 * compressed runs. These are timings and battery levels only: nothing about the user's data.
 */

/** One battery reading. [chargeMicroAh] is null when the phone does not report its charge. */
class BatteryReading(val percent: Int, val chargeMicroAh: Long?, val charging: Boolean)

/** What one run cost on this phone. */
class DeviceCost(
    val runSeconds: Double,
    /** Building the 12 features (or more) from 1 second of sensor data. */
    val featureMsPerWindow: Double,
    /** Training the local-only model from zero (all its epochs). */
    val baselineTrainMs: Double,
    /** One federated round of local training (average over the rounds). */
    val trainMsPerRound: Double,
    /** Classifying one window with the three models. */
    val predictMsPerWindow: Double,
    val bytesSent: Long,
    val bytesReceived: Long,
    val rounds: Int,
    val batteryStartPercent: Int?,
    val batteryEndPercent: Int?,
    /** From the battery's charge counter; null if the phone does not report it. */
    val batteryUsedMah: Double?,
    /** True if the phone was charging at the start or the end: the battery numbers then mean nothing. */
    val charging: Boolean,
) {
    /** The upload to POST /device_cost (the server's allow-list has exactly these fields). */
    fun toMap(clientId: String, runId: String): Map<String, Any?> = linkedMapOf(
        "client_id" to clientId, "run_id" to runId,
        "run_s" to round2(runSeconds),
        "feature_ms_per_window" to round3(featureMsPerWindow),
        "baseline_train_ms" to baselineTrainMs,
        "train_ms_per_round" to round2(trainMsPerRound),
        "predict_ms_per_window" to round3(predictMsPerWindow),
        "bytes_sent" to bytesSent, "bytes_received" to bytesReceived, "rounds" to rounds,
        "battery_start_pct" to batteryStartPercent, "battery_end_pct" to batteryEndPercent,
        "battery_used_mah" to batteryUsedMah?.let { round2(it) }, "charging" to charging,
    )

    /** A few lines for the phone's results screen. */
    fun describe(): String {
        val sb = StringBuilder()
        sb.append(String.format(Locale.US, "Whole run: %.0f s, %d rounds\n", runSeconds, rounds))
        sb.append(String.format(Locale.US, "Features for one window: %.3f ms\n", featureMsPerWindow))
        sb.append(String.format(Locale.US, "Local-only model, trained from zero: %.1f ms\n", baselineTrainMs))
        sb.append(String.format(Locale.US, "Training per federated round: %.1f ms\n", trainMsPerRound))
        sb.append(String.format(Locale.US, "Prediction, 3 models, one window: %.3f ms\n", predictMsPerWindow))
        sb.append(String.format(Locale.US, "Data sent: %s, received: %s\n", kb(bytesSent), kb(bytesReceived)))
        sb.append("Battery: " + when {
            charging -> "not measured (the phone was charging; unplug it to measure)"
            batteryUsedMah != null -> String.format(Locale.US, "%.1f mAh used (%d%% -> %d%%)", batteryUsedMah,
                batteryStartPercent ?: 0, batteryEndPercent ?: 0)
            batteryStartPercent != null -> "${batteryStartPercent}% -> ${batteryEndPercent}%"
            else -> "not available on this phone"
        })
        return sb.toString()
    }

    private fun kb(bytes: Long) = String.format(Locale.US, "%.1f KB", bytes / 1000.0)
    private fun round2(v: Double) = Math.round(v * 100) / 100.0
    private fun round3(v: Double) = Math.round(v * 1000) / 1000.0
}

/**
 * A stopwatch for one run: [start] when the run begins, [addRound] after each round of local training,
 * [finish] at the end. [battery] reads the phone's battery (null in JVM tests).
 */
class CostTracker(private val transport: Transport, private val battery: () -> BatteryReading?) {
    private var startMs = 0L
    private var sentAtStart = 0L
    private var receivedAtStart = 0L
    private var batteryAtStart: BatteryReading? = null
    private var baselineMs = 0.0
    private var roundMs = ArrayList<Double>()

    fun start() {
        startMs = System.currentTimeMillis()
        sentAtStart = transport.bytesSent
        receivedAtStart = transport.bytesReceived
        batteryAtStart = battery()
        baselineMs = 0.0
        roundMs = ArrayList()
    }

    fun baselineTrained(ms: Double) { baselineMs = ms }

    fun addRound(ms: Double) { roundMs.add(ms) }

    fun finish(featureMsPerWindow: Double, predictMsPerWindow: Double): DeviceCost {
        val end = battery()
        val start = batteryAtStart
        val usedMah = if (start?.chargeMicroAh != null && end?.chargeMicroAh != null)
            (start.chargeMicroAh - end.chargeMicroAh) / 1000.0 else null
        return DeviceCost(
            runSeconds = (System.currentTimeMillis() - startMs) / 1000.0,
            featureMsPerWindow = featureMsPerWindow,
            baselineTrainMs = baselineMs,
            trainMsPerRound = if (roundMs.isEmpty()) 0.0 else roundMs.average(),
            predictMsPerWindow = predictMsPerWindow,
            bytesSent = transport.bytesSent - sentAtStart,
            bytesReceived = transport.bytesReceived - receivedAtStart,
            rounds = roundMs.size,
            batteryStartPercent = start?.percent,
            batteryEndPercent = end?.percent,
            batteryUsedMah = usedMah,
            charging = (start?.charging ?: false) || (end?.charging ?: false),
        )
    }

    companion object {
        /**
         * How long the three models take to classify one window: runs them over [windows] (repeated
         * until at least [minPredictions] windows were classified) and divides the time.
         */
        fun timePredictions(models: List<SoftmaxModel>, windows: List<DoubleArray>, minPredictions: Int = 300): Double {
            if (models.isEmpty() || windows.isEmpty()) return 0.0
            var done = 0
            var sink = 0
            val t0 = System.nanoTime()
            while (done < minPredictions) {
                for (x in windows) {
                    for (m in models) sink += m.predict(x)
                    done++
                }
            }
            val ms = (System.nanoTime() - t0) / 1e6
            if (sink == Int.MIN_VALUE) println()   // keeps the work from being optimised away
            return ms / done
        }
    }
}
