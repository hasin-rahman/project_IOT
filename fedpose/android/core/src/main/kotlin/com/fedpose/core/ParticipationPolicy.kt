package com.fedpose.core

/**
 * Adaptive participation (client side): skip an upload when the local update is small.
 * The server overrides this with `must_participate` after too many consecutive skips so that
 * both clients keep contributing.
 */
class ParticipationPolicy(val enabled: Boolean = false, val threshold: Double = 0.0) {
    fun shouldSkip(updateNorm: Double, mustParticipate: Boolean): Boolean =
        enabled && !mustParticipate && updateNorm < threshold
}
