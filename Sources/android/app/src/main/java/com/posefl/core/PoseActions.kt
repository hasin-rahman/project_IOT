package com.posefl.core

/**
 * Pose actions: the phone DOES something when it recognises a pose (sense -> decide -> act).
 *
 * Two actions use client A's poses and two use client B's, so a local-only model can trigger only
 * the actions of its own phone's poses, while the federated global model can trigger all four.
 * Portrait Up (how people normally hold a phone) and Landscape Right have no action.
 */
enum class PoseAction(val pose: Pose, val title: String, val effect: String) {
    DO_NOT_DISTURB(Pose.SCREEN_DOWN, "Do Not Disturb", "notifications are silenced"),
    DESK_CLOCK(Pose.SCREEN_UP, "Desk clock", "a big clock fills the screen"),
    FLASHLIGHT(Pose.PORTRAIT_DOWN, "Flashlight", "the flashlight turns on"),
    TIMER(Pose.LANDSCAPE_LEFT, "Timer", "a timer starts counting");

    /** Which client trains on this action's pose ("A" or "B"). */
    val client: String get() = if (pose in Pose.trainPosesFor("A")) "A" else "B"

    companion object {
        /** The action for [pose], or null if that pose has none. */
        fun forPose(pose: Pose?): PoseAction? = entries.firstOrNull { it.pose == pose }
    }
}

/**
 * Decides when an action switches on and off, from the live predictions of ONE model.
 * Call [update] every time the model gives a new answer (about 5 times a second).
 *
 *  - ON:  the model has said the action's pose, with at least [minConfidence], for [holdOnMs] in a row (3 s)
 *  - OFF: the model has said something else (or was unsure) for [releaseMs] in a row (1 s)
 *
 * Waiting 3 s before switching on means picking the phone up or putting it down never triggers anything.
 */
class PoseActionTrigger(
    private val holdOnMs: Long = 3000,
    private val releaseMs: Long = 1000,
    /** How sure the model must be (its top probability). 75%: well above guessing (17%), and the
     *  3-second hold already stops accidental triggers. */
    val minConfidence: Double = 0.75,
) {
    /** The action that is on right now (null = none). */
    var active: PoseAction? = null
        private set

    /** The action whose pose is being held but is not on yet, and since when. */
    var holding: PoseAction? = null
        private set
    private var holdingSince = 0L
    private var leftSince = 0L

    /** What happened in one [update]. */
    sealed class Change {
        class Started(val action: PoseAction) : Change()
        class Stopped(val action: PoseAction) : Change()
    }

    /** [pose]/[confidence]: the model's current answer. Returns what changed, if anything. */
    fun update(pose: Pose, confidence: Double, nowMs: Long): Change? {
        val seen = if (confidence >= minConfidence) pose else null

        // 1. an action is on: switch it off once the phone has left its pose for releaseMs
        val on = active
        if (on != null) {
            if (seen == on.pose) { leftSince = 0L; return null }
            if (leftSince == 0L) leftSince = nowMs
            if (nowMs - leftSince < releaseMs) return null
            active = null
            leftSince = 0L
            holding = null
            return Change.Stopped(on)
        }

        // 2. nothing is on: count how long the same action's pose has been held
        val action = PoseAction.forPose(seen)
        if (action == null) { holding = null; return null }
        if (action != holding) { holding = action; holdingSince = nowMs; return null }
        if (nowMs - holdingSince < holdOnMs) return null
        active = action
        holding = null
        return Change.Started(action)
    }

    /** 0.0 .. 1.0: how far the held pose is towards switching its action on (for the progress bar). */
    fun holdProgress(nowMs: Long): Double =
        if (holding == null) 0.0 else ((nowMs - holdingSince).toDouble() / holdOnMs).coerceIn(0.0, 1.0)

    /** Switch everything off at once (e.g. when the screen is left). Returns the action that was on. */
    fun reset(): PoseAction? {
        val was = active
        active = null; holding = null; leftSince = 0L
        return was
    }
}
