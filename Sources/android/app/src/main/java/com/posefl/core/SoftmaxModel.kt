package com.posefl.core

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max

/**
 * The classifier: softmax regression (a single linear layer followed by softmax).
 *
 *   score of pose c  =  b[c] + W[c][0]*x[0] + ... + W[c][F-1]*x[F-1]
 *   probabilities    =  softmax(scores)
 *   prediction       =  the pose with the highest probability
 *
 * Parameters: W has nClasses x nFeatures numbers (stored row by row in one array) and b has nClasses.
 * Baseline: 12 features -> 6*12 + 6 = 78 parameters. This is ALL that a phone sends to the server.
 * It is trained with mini-batch SGD. Mirrors reference/posefl_ref/model.py.
 */
class SoftmaxModel(val nFeatures: Int, val nClasses: Int = Pose.COUNT) {
    /** Weights. A new model starts with every weight at 0 ("from scratch"). */
    val W = DoubleArray(nClasses * nFeatures)
    /** One bias per pose, also starting at 0. */
    val b = DoubleArray(nClasses)

    val numParams: Int get() = W.size + b.size

    fun copy(): SoftmaxModel {
        val m = SoftmaxModel(nFeatures, nClasses)
        W.copyInto(m.W); b.copyInto(m.b)
        return m
    }

    fun setFrom(other: SoftmaxModel) {
        require(other.nFeatures == nFeatures && other.nClasses == nClasses) { "model shape mismatch" }
        other.W.copyInto(W); other.b.copyInto(b)
    }

    // ------------------------------------------------------------------ prediction

    /** The probability of each pose for one feature vector [x]. The six numbers add up to 1. */
    fun predictProba(x: DoubleArray): DoubleArray {
        // 1) one score per pose: bias + weights . features
        val z = DoubleArray(nClasses)
        var zMax = Double.NEGATIVE_INFINITY
        for (c in 0 until nClasses) {
            var s = b[c]
            val off = c * nFeatures
            for (f in 0 until nFeatures) s += W[off + f] * x[f]
            z[c] = s
            zMax = max(zMax, s)
        }
        // 2) softmax: exp() of each score, divided by the sum (subtracting the max avoids overflow)
        var total = 0.0
        for (c in 0 until nClasses) { z[c] = exp(z[c] - zMax); total += z[c] }
        for (c in 0 until nClasses) z[c] /= total
        return z
    }

    /** The index of the most likely pose for [x]. */
    fun predict(x: DoubleArray): Int = argmax(predictProba(x))

    data class Eval(val loss: Double, val accuracy: Double)

    /** Average cross-entropy loss and accuracy on a labelled dataset. */
    fun evaluate(X: List<DoubleArray>, y: IntArray): Eval {
        if (X.isEmpty()) return Eval(0.0, 0.0)
        var loss = 0.0
        var correct = 0
        for (i in X.indices) {
            val p = predictProba(X[i])
            loss -= ln(max(p[y[i]], 1e-12))
            if (argmax(p) == y[i]) correct++
        }
        return Eval(loss / X.size, correct.toDouble() / X.size)
    }

    // ------------------------------------------------------------------ training

    /**
     * Mini-batch SGD. One epoch = one pass over all the training examples, in batches of [batchSize].
     * [l2] is a small penalty on large weights (on W, not b). [rng] shuffles the examples each epoch;
     * null means no shuffling (deterministic, used by tests).
     */
    fun train(
        X: List<DoubleArray>, y: IntArray,
        epochs: Int = 1, lr: Double = 0.1, batchSize: Int = 16, l2: Double = 1e-4,
        rng: java.util.Random? = null,
    ) {
        val n = X.size
        if (n == 0) return
        val order = IntArray(n) { it }
        repeat(epochs) {
            if (rng != null) shuffle(order, rng)
            var start = 0
            while (start < n) {
                val end = minOf(start + batchSize, n)
                sgdStep(X, y, order, start, end, lr, l2)
                start = end
            }
        }
    }

    /**
     * One SGD step on the examples order[start until end]:
     *   1) for every example: error = predicted probabilities - correct answer (1 for the true pose, 0 otherwise)
     *   2) gradient = average of error * features over the batch
     *   3) move every weight a small step (the learning rate [lr]) against the gradient
     */
    private fun sgdStep(X: List<DoubleArray>, y: IntArray, order: IntArray, start: Int, end: Int, lr: Double, l2: Double) {
        val gW = DoubleArray(W.size)
        val gb = DoubleArray(nClasses)
        for (k in start until end) {
            val x = X[order[k]]
            val error = predictProba(x)
            error[y[order[k]]] -= 1.0
            for (c in 0 until nClasses) {
                gb[c] += error[c]
                val off = c * nFeatures
                for (f in 0 until nFeatures) gW[off + f] += error[c] * x[f]
            }
        }
        val m = (end - start).toDouble()
        for (i in W.indices) W[i] -= lr * (gW[i] / m + l2 * W[i])
        for (c in 0 until nClasses) b[c] -= lr * gb[c] / m
    }

    /** Fisher-Yates shuffle, so every epoch sees the examples in a different order. */
    private fun shuffle(order: IntArray, rng: java.util.Random) {
        for (i in order.size - 1 downTo 1) {
            val j = rng.nextInt(i + 1)
            val tmp = order[i]; order[i] = order[j]; order[j] = tmp
        }
    }

    companion object {
        /** Index of the largest value (the first one if there is a tie). */
        fun argmax(values: DoubleArray): Int {
            var best = 0
            for (c in 1 until values.size) if (values[c] > values[best]) best = c
            return best
        }

        /**
         * Federated Averaging: every parameter of the new model is the average of the clients'
         * parameters, weighted by how many training examples each client has.
         * The server does exactly this in Python (server/fedavg.py); this copy is used by tests.
         */
        fun average(models: List<SoftmaxModel>, counts: List<Int>): SoftmaxModel {
            require(models.isNotEmpty() && models.size == counts.size)
            val total = counts.sum().toDouble()
            val out = SoftmaxModel(models[0].nFeatures, models[0].nClasses)
            for ((m, c) in models.zip(counts)) {
                val w = c / total
                for (i in out.W.indices) out.W[i] += w * m.W[i]
                for (i in out.b.indices) out.b[i] += w * m.b[i]
            }
            return out
        }
    }
}
