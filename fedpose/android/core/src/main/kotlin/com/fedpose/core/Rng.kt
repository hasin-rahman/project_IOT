package com.fedpose.core

private val GOLDEN: Long = 0x9E3779B97F4A7C15uL.toLong()
private val MIX1: Long = 0xBF58476D1CE4E5B9uL.toLong()
private val MIX2: Long = 0x94D049BB133111EBuL.toLong()

/** SplitMix64, identical to python/fedpose/rng.py so shuffles match bit for bit. */
class SplitMix64(seed: Long) {
    private var s: Long = seed

    fun nextU64(): Long {
        s += GOLDEN
        var z = s
        z = (z xor (z ushr 30)) * MIX1
        z = (z xor (z ushr 27)) * MIX2
        return z xor (z ushr 31)
    }

    fun nextInt(bound: Int): Int = ((nextU64() ushr 1) % bound.toLong()).toInt()

    /** Uniform in [0, 1) with 53 bits. */
    fun nextDouble(): Double = (nextU64() ushr 11).toDouble() / 9007199254740992.0

    /** Fisher-Yates from the end. */
    fun shuffle(a: IntArray) {
        for (i in a.size - 1 downTo 1) {
            val j = nextInt(i + 1)
            val t = a[i]; a[i] = a[j]; a[j] = t
        }
    }
}
