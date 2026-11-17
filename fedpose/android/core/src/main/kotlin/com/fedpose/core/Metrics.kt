package com.fedpose.core

/** Confusion matrix based evaluation. Twin of python/fedpose/metrics.py. */
class Eval(val confusion: Array<IntArray>) {
    private val k = Constants.N_CLASSES
    val n: Int = confusion.sumOf { it.sum() }
    val accuracy: Double
        get() = if (n == 0) 0.0 else (0 until k).sumOf { confusion[it][it] }.toDouble() / n
    val perClassRecall: List<Double>
        get() = (0 until k).map { c -> val s = confusion[c].sum(); if (s == 0) 0.0 else confusion[c][c].toDouble() / s }
    val perClassF1: List<Double>
        get() = (0 until k).map { c ->
            val tp = confusion[c][c].toDouble()
            val sup = confusion[c].sum().toDouble()
            val pred = (0 until k).sumOf { confusion[it][c] }.toDouble()
            val r = if (sup == 0.0) 0.0 else tp / sup
            val p = if (pred == 0.0) 0.0 else tp / pred
            if (p + r == 0.0) 0.0 else 2 * p * r / (p + r)
        }
    val macroF1: Double get() = perClassF1.average()

    fun toMap(): Map<String, Any?> = linkedMapOf(
        "accuracy" to accuracy, "macro_f1" to macroF1, "n" to n,
        "per_class_recall" to perClassRecall, "per_class_f1" to perClassF1,
        "confusion" to confusion.map { it.toList() },
    )

    companion object {
        fun of(model: SoftmaxModel, w: DoubleArray, d: LocalData): Eval {
            val cm = Array(Constants.N_CLASSES) { IntArray(Constants.N_CLASSES) }
            for (i in 0 until d.size) cm[d.y[i]][model.predict(w, d.X[i])]++
            return Eval(cm)
        }
    }
}
