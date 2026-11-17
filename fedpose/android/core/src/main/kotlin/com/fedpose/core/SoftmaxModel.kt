package com.fedpose.core

import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min

data class TrainConfig(
    val epochs: Int = 2,
    val lr: Double = 0.3,
    val batchSize: Int = 32,
    val l2: Double = 0.0,
    /** FedProx proximal weight; 0 = plain FedAvg local training. */
    val mu: Double = 0.0,
)

/** Labelled windows held ON THE PHONE. These never leave the device. */
class LocalData(val X: List<DoubleArray>, val y: IntArray) {
    val size: Int get() = y.size
}

/**
 * Softmax regression, 6 x nf weights + 6 biases (78 parameters for the 12-feature baseline).
 * Parameter layout: W[0][0..nf-1], ..., W[5][0..nf-1], b[0..5]. Twin of python/fedpose/model.py.
 */
class SoftmaxModel(val nf: Int) {
    val nParams: Int = Constants.nParams(nf)

    fun logits(w: DoubleArray, x: DoubleArray, out: DoubleArray = DoubleArray(Constants.N_CLASSES)): DoubleArray {
        val bOff = Constants.N_CLASSES * nf
        for (k in 0 until Constants.N_CLASSES) {
            var z = w[bOff + k]
            val r = k * nf
            for (j in 0 until nf) z += w[r + j] * x[j]
            out[k] = z
        }
        return out
    }

    fun softmaxInPlace(z: DoubleArray) {
        var m = z[0]
        for (v in z) if (v > m) m = v
        var s = 0.0
        for (k in z.indices) { z[k] = exp(z[k] - m); s += z[k] }
        for (k in z.indices) z[k] /= s
    }

    fun proba(w: DoubleArray, x: DoubleArray): DoubleArray = logits(w, x).also { softmaxInPlace(it) }

    fun predict(w: DoubleArray, x: DoubleArray): Int {
        val z = logits(w, x)
        var best = 0
        for (k in 1 until z.size) if (z[k] > z[best]) best = k  // first max wins, like numpy argmax
        return best
    }

    /** Returns (cross-entropy loss, accuracy). */
    fun lossAcc(w: DoubleArray, d: LocalData): Pair<Double, Double> {
        if (d.size == 0) return 0.0 to 0.0
        var loss = 0.0
        var ok = 0
        for (i in 0 until d.size) {
            val p = proba(w, d.X[i])
            loss -= ln(maxOf(p[d.y[i]], 1e-12))
            var best = 0
            for (k in 1 until p.size) if (p[k] > p[best]) best = k
            if (best == d.y[i]) ok++
        }
        return loss / d.size to ok.toDouble() / d.size
    }

    /** Minibatch SGD on local data only. `wRef` is the global model for the FedProx term. */
    fun train(w0: DoubleArray, d: LocalData, cfg: TrainConfig, seed: Long, wRef: DoubleArray = w0): DoubleArray {
        val w = w0.copyOf()
        val n = d.size
        if (n == 0) return w
        val rng = SplitMix64(seed)
        val idx = IntArray(n) { it }
        val bOff = Constants.N_CLASSES * nf
        val gW = DoubleArray(Constants.N_CLASSES * nf)
        val gb = DoubleArray(Constants.N_CLASSES)
        val p = DoubleArray(Constants.N_CLASSES)
        repeat(cfg.epochs) {
            rng.shuffle(idx)
            var s = 0
            while (s < n) {
                val e = min(s + cfg.batchSize, n)
                val m = (e - s).toDouble()
                gW.fill(0.0); gb.fill(0.0)
                for (q in s until e) {
                    val i = idx[q]
                    val x = d.X[i]
                    logits(w, x, p)
                    softmaxInPlace(p)
                    p[d.y[i]] -= 1.0
                    for (k in 0 until Constants.N_CLASSES) {
                        gb[k] += p[k]
                        val r = k * nf
                        for (j in 0 until nf) gW[r + j] += p[k] * x[j]
                    }
                }
                for (k in 0 until Constants.N_CLASSES) {
                    val r = k * nf
                    for (j in 0 until nf) {
                        var g = gW[r + j] / m + cfg.l2 * w[r + j]
                        if (cfg.mu > 0) g += cfg.mu * (w[r + j] - wRef[r + j])
                        w[r + j] -= cfg.lr * g
                    }
                    var g = gb[k] / m
                    if (cfg.mu > 0) g += cfg.mu * (w[bOff + k] - wRef[bOff + k])
                    w[bOff + k] -= cfg.lr * g
                }
                s = e
            }
        }
        return w
    }

    companion object {
        /** Zeros by default; otherwise uniform in [-scale, scale] from SplitMix64 (same as Python). */
        fun initWeights(nf: Int, seed: Long = 0, scale: Double = 0.0): DoubleArray {
            val w = DoubleArray(Constants.nParams(nf))
            if (scale > 0) {
                val rng = SplitMix64(seed)
                for (i in w.indices) w[i] = (rng.nextDouble() - 0.5) * 2.0 * scale
            }
            return w
        }
    }
}
