package com.posefl.core

/**
 * Scores a model: compares the true poses with the predicted poses.
 * Gives accuracy, per-pose recall and F1, macro-F1 and the confusion matrix.
 * Mirrors reference/posefl_ref/metrics.py.
 */
class Metrics(yTrue: IntArray, yPred: IntArray, val nClasses: Int = Pose.COUNT) {
    /** confusion[true pose][predicted pose] = how many times that happened. */
    val confusion = Array(nClasses) { IntArray(nClasses) }
    /** Share of all samples that were predicted correctly. */
    val accuracy: Double
    /** For each pose: of the samples that really were this pose, the share predicted correctly. */
    val perClassRecall = DoubleArray(nClasses)
    val perClassF1 = DoubleArray(nClasses)
    /** Average F1 over the poses that appear in the data (every pose counts equally). */
    val macroF1: Double
    val total: Int = yTrue.size

    init {
        require(yTrue.size == yPred.size)
        for (i in yTrue.indices) confusion[yTrue[i]][yPred[i]]++
        var correct = 0
        var f1Sum = 0.0
        var present = 0
        for (c in 0 until nClasses) {
            val tp = confusion[c][c].toDouble()
            val support = confusion[c].sum()
            var predicted = 0
            for (r in 0 until nClasses) predicted += confusion[r][c]
            correct += confusion[c][c]
            val recall = if (support > 0) tp / support else 0.0
            val precision = if (predicted > 0) tp / predicted else 0.0
            perClassRecall[c] = recall
            perClassF1[c] = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0
            if (support > 0) { f1Sum += perClassF1[c]; present++ }
        }
        accuracy = if (total > 0) correct.toDouble() / total else 0.0
        macroF1 = if (present > 0) f1Sum / present else 0.0
    }

    /** How many samples truly belong to one of [poses]. */
    fun countOn(poses: List<Pose>): Int = poses.sumOf { confusion[it.index].sum() }

    /** How many of those samples were predicted correctly. */
    fun correctOn(poses: List<Pose>): Int = poses.sumOf { confusion[it.index][it.index] }

    /**
     * Accuracy on only the samples whose TRUE pose is in [poses], or null if there are none.
     * Example: a local-only model of client A should be high on A's three poses and
     * about 0 on B's three poses, because it never saw them in training.
     */
    fun accuracyOn(poses: List<Pose>): Double? {
        val n = countOn(poses)
        return if (n == 0) null else correctOn(poses).toDouble() / n
    }
}
