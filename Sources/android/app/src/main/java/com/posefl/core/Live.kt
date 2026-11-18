package com.posefl.core

import kotlin.math.sqrt

/**
 * What one model thinks about the latest 1-second window.
 *  - [windowBest]: its answer from this ONE window alone (used when a test placement is captured)
 *  - [best]: its answer after averaging the last few windows (steadier, used for the live display)
 *  - [pose]: same as [best], but null when the confidence is too low (shown as "uncertain")
 */
class LivePrediction(
    val modelName: String,
    val pose: Pose?,
    val best: Pose,
    val confidence: Double,
    val probs: DoubleArray,
    val windowBest: Pose,
)

class LiveFrame(
    val predictions: List<LivePrediction>,
    /** The phone is moving (the acceleration is not steady). */
    val moving: Boolean,
    /** Not enough samples in the window yet. */
    val warmingUp: Boolean,
) {
    /** Each model's answer for this one window: model name -> pose. */
    fun windowAnswers(): Map<String, Pose> = predictions.associate { it.modelName to it.windowBest }
}

/**
 * Live pose prediction, using the same 1-second window and the same features as training.
 * All three models get exactly the same window, so their answers can be compared fairly.
 *
 * For the live display, each model's probabilities are averaged over the last [smoothing] windows
 * so one noisy window cannot flip the answer. Below [minConfidence] the display says "uncertain"
 * (with six poses, pure guessing is 17%).
 */
class LivePredictor(
    private val windowMs: Double = 1000.0,
    private val smoothing: Int = 4,
    private val minConfidence: Double = 0.4,
    private val movingThresholdG: Double = 0.06,
    private val minSamples: Int = 25,
) {
    /** The last few probability vectors of each model (for smoothing). */
    private val history = HashMap<String, ArrayDeque<DoubleArray>>()

    fun reset() = history.clear()

    /** [rows] are recent raw sensor rows (SensorCsv layout, time in ms, oldest first). */
    fun update(rows: List<DoubleArray>, models: List<StoredModel>): LiveFrame {
        val window = latestWindow(rows) ?: return LiveFrame(emptyList(), moving = false, warmingUp = true)
        val featuresBySet = HashMap<String, DoubleArray>()   // computed once per feature set, shared by the models
        val predictions = models.map { stored ->
            val x = featuresBySet.getOrPut(stored.featureSet) { FeatureExtractor.windowFeatures(rows, window, stored.featureSet) }
            predictOne(stored, x)
        }
        return LiveFrame(predictions, moving = isMoving(rows, window), warmingUp = false)
    }

    /** The row indexes of the most recent [windowMs] of data, or null if there are too few samples. */
    private fun latestWindow(rows: List<DoubleArray>): IntRange? =
        if (rows.isEmpty()) null else windowEndingAt(rows, rows.last()[0])

    /** The row indexes of the [windowMs] of data ending at time [endMs], or null if there are too few samples. */
    private fun windowEndingAt(rows: List<DoubleArray>, endMs: Double): IntRange? {
        var last = rows.size
        while (last > 0 && rows[last - 1][0] > endMs) last--
        var first = last
        while (first > 0 && rows[first - 1][0] > endMs - windowMs) first--
        return if (last - first < minSamples) null else first until last
    }

    /**
     * The blind-test answer of each model (Hasin's and Shiv's ideas combined): the app captures only
     * once the phone has been still (see PredictActivity), and then, instead of trusting one window,
     * averages each model's probabilities over the last [windows] 1-second windows, spaced [stepMs]
     * apart (by default the newest window and the one ending 0.5 s earlier). One bump in one window
     * can no longer decide the answer on its own. Returns an empty map if there is not enough data.
     */
    fun captureAnswers(rows: List<DoubleArray>, models: List<StoredModel>, windows: Int = 2, stepMs: Double = 500.0): Map<String, Pose> {
        if (rows.isEmpty()) return emptyMap()
        val newest = rows.last()[0]
        val ranges = (0 until windows).mapNotNull { k -> windowEndingAt(rows, newest - k * stepMs) }
        if (ranges.isEmpty()) return emptyMap()
        return models.associate { stored ->
            val average = DoubleArray(Pose.COUNT)
            for (range in ranges) {
                val p = stored.model.predictProba(FeatureExtractor.windowFeatures(rows, range, stored.featureSet))
                for (c in average.indices) average[c] += p[c] / ranges.size
            }
            stored.name to Pose.fromIndex(SoftmaxModel.argmax(average))
        }
    }

    /** Moving = the size of the acceleration (in g) varies by more than [movingThresholdG] within the window. */
    private fun isMoving(rows: List<DoubleArray>, window: IntRange): Boolean {
        var sum = 0.0
        var sumSquares = 0.0
        for (i in window) {
            val r = rows[i]
            val magnitude = sqrt(r[1] * r[1] + r[2] * r[2] + r[3] * r[3]) / FeatureSets.G
            sum += magnitude
            sumSquares += magnitude * magnitude
        }
        val n = window.count()
        val mean = sum / n
        val std = sqrt(maxOf(0.0, sumSquares / n - mean * mean))
        return std > movingThresholdG
    }

    /** Runs one model on the window's features and smooths its answer over the last few windows. */
    private fun predictOne(stored: StoredModel, x: DoubleArray): LivePrediction {
        val probs = stored.model.predictProba(x)
        val recent = history.getOrPut(stored.name) { ArrayDeque() }
        recent.addLast(probs)
        while (recent.size > smoothing) recent.removeFirst()

        val average = DoubleArray(probs.size)
        for (p in recent) for (c in average.indices) average[c] += p[c] / recent.size
        val best = SoftmaxModel.argmax(average)
        val confident = average[best] >= minConfidence
        return LivePrediction(
            modelName = stored.name,
            pose = if (confident) Pose.fromIndex(best) else null,
            best = Pose.fromIndex(best),
            confidence = average[best],
            probs = average,
            windowBest = Pose.fromIndex(SoftmaxModel.argmax(probs)),
        )
    }
}

/** Names used for the three models everywhere in the app. */
object ModelNames {
    const val GLOBAL = "global"
    const val LOCAL_A = "local_A"
    const val LOCAL_B = "local_B"
    /** Display order: the federated model first, then the two local-only baselines. */
    val ALL = listOf(GLOBAL, LOCAL_A, LOCAL_B)

    fun label(model: String) = when (model) {
        GLOBAL -> "Global"
        LOCAL_A -> "Local A"
        LOCAL_B -> "Local B"
        else -> model
    }
}

// =====================================================================================
// Blind test (demo day): the judges choose the pose, the app only records what the models said
// =====================================================================================

/** One placement of the phone in the blind test: the answer of each model for one window. */
class Placement(val number: Int, val answers: Map<String, Pose>, val still: Boolean, val timeMs: Long)

/**
 * The record of a blind test. The judges place the phone 36 times (each pose 6 times) following
 * their pose cards. The app does not know the true pose; it shows and logs the three predictions.
 */
class BlindTestLog(val plannedPlacements: Int = 36) {
    val placements = ArrayList<Placement>()

    /** The number the next placement will get (1, 2, 3, ...). */
    val nextNumber: Int get() = placements.size + 1

    fun add(answers: Map<String, Pose>, still: Boolean): Placement {
        val p = Placement(nextNumber, answers, still, System.currentTimeMillis())
        placements.add(p)
        return p
    }

    fun undoLast(): Placement? = if (placements.isEmpty()) null else placements.removeAt(placements.size - 1)

    fun toCsv(): String {
        val sb = StringBuilder("placement,time_ms," + ModelNames.ALL.joinToString(",") + ",still\n")
        for (p in placements) {
            sb.append(p.number).append(',').append(p.timeMs)
            for (m in ModelNames.ALL) sb.append(',').append(p.answers[m]?.key ?: "none")
            sb.append(',').append(p.still).append('\n')
        }
        return sb.toString()
    }
}

// =====================================================================================
// Practice test: the app asks for a pose, so it knows the truth and can keep score
// =====================================================================================

/** One practice trial: the pose asked for and what each model answered. */
class Trial(val target: Pose, val answers: Map<String, Pose?>, val tester: String, val timeMs: Long)

/**
 * Practice mode for rehearsing the blind test. The app asks for poses in a shuffled order
 * (each pose once per block of six, so 36 trials = every pose 6 times, like the real test)
 * and scores the answers the way the demo-day rubric does (see [rubricPoints]).
 */
class LiveTest(private val modelNames: List<String>, seed: Long = System.currentTimeMillis()) {
    private val rng = java.util.Random(seed)
    private val queue = ArrayDeque<Pose>()
    val trials = ArrayList<Trial>()

    fun nextTarget(): Pose {
        if (queue.isEmpty()) queue.addAll(Pose.entries.shuffled(rng))
        return queue.first()
    }

    fun record(target: Pose, answers: Map<String, Pose?>, tester: String) {
        trials.add(Trial(target, answers, tester, System.currentTimeMillis()))
        if (queue.firstOrNull() == target) queue.removeFirst()
    }

    fun undoLast(): Trial? = if (trials.isEmpty()) null else trials.removeAt(trials.size - 1)

    /** Trials whose target pose is one of [poses] (default: all six). */
    fun countOn(poses: List<Pose> = Pose.entries): Int = trials.count { it.target in poses }

    /** How many of those trials [model] got right. */
    fun correctOn(model: String, poses: List<Pose> = Pose.entries): Int =
        trials.count { it.target in poses && it.answers[model] == it.target }

    fun correct(model: String) = correctOn(model)

    fun accuracy(model: String, poses: List<Pose> = Pose.entries): Double {
        val n = countOn(poses)
        return if (n == 0) 0.0 else correctOn(model, poses).toDouble() / n
    }

    /**
     * Demo-day rubric, part A (40 points):
     *   A1 = 30 x global model accuracy over ALL placements
     *   A2 =  5 x local model A accuracy on client A's three poses
     *   A3 =  5 x local model B accuracy on client B's three poses
     */
    fun rubricPoints(): Double =
        30 * accuracy(ModelNames.GLOBAL) +
            5 * accuracy(ModelNames.LOCAL_A, Pose.trainPosesFor("A")) +
            5 * accuracy(ModelNames.LOCAL_B, Pose.trainPosesFor("B"))

    /** Metrics for one model (a missing answer is scored as wrong). */
    fun metrics(model: String): Metrics {
        val yTrue = IntArray(trials.size) { trials[it].target.index }
        val yPred = IntArray(trials.size) { trials[it].answers[model]?.index ?: wrongIndex(trials[it].target) }
        return Metrics(yTrue, yPred)
    }

    private fun wrongIndex(target: Pose) = (target.index + 3) % Pose.COUNT

    fun summary(): String = if (trials.isEmpty()) "No trials yet" else modelNames.joinToString("   ") { m ->
        "${label(m)} ${correct(m)}/${trials.size} (${(accuracy(m) * 100).toInt()}%)"
    }

    fun toCsv(): String {
        val sb = StringBuilder("time_ms,tester,target," + modelNames.joinToString(",") + "\n")
        for (t in trials) {
            sb.append(t.timeMs).append(',').append(t.tester).append(',').append(t.target.key)
            for (m in modelNames) sb.append(',').append(t.answers[m]?.key ?: "none")
            sb.append('\n')
        }
        return sb.toString()
    }

    companion object {
        fun label(model: String) = ModelNames.label(model)
    }
}
