package com.posefl.app

import android.app.Activity
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.media.ToneGenerator
import android.os.VibrationEffect
import android.os.Vibrator
import android.view.Gravity
import android.view.View
import android.view.ViewGroup
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView

/** Small helpers for building the training and live screens in code (no XML layouts to keep in sync). */
internal fun Activity.dp(v: Int): Int = (v * resources.displayMetrics.density + 0.5f).toInt()

internal fun Activity.vbox(padding: Int = 0): LinearLayout = LinearLayout(this).apply {
    orientation = LinearLayout.VERTICAL
    if (padding > 0) setPadding(dp(padding), dp(padding), dp(padding), dp(padding))
}

internal fun Activity.hbox(): LinearLayout = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }

internal fun Activity.label(text: String = "", sizeSp: Float = 15f, bold: Boolean = false, colorRes: Int = 0, mono: Boolean = false): TextView =
    TextView(this).apply {
        this.text = text
        textSize = sizeSp
        if (bold) setTypeface(typeface, Typeface.BOLD)
        if (mono) typeface = Typeface.MONOSPACE
        if (colorRes != 0) setTextColor(getColor(colorRes))
    }

internal fun Activity.button(text: String, onClick: () -> Unit): Button = Button(this).apply {
    this.text = text
    setOnClickListener { onClick() }
}

internal fun Activity.card(colorRes: Int, radiusDp: Int = 12): GradientDrawable = GradientDrawable().apply {
    setColor(getColor(colorRes))
    cornerRadius = dp(radiusDp).toFloat()
}

/** Adds [child] to this LinearLayout, full width, with a top margin. */
internal fun LinearLayout.add(child: View, topMarginPx: Int = 0, weight: Float = 0f): LinearLayout {
    val lp = if (orientation == LinearLayout.VERTICAL)
        LinearLayout.LayoutParams(ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT)
    else
        LinearLayout.LayoutParams(if (weight > 0f) 0 else ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT, weight)
    lp.topMargin = topMarginPx
    addView(child, lp)
    return this
}

internal fun TextView.center(): TextView { gravity = Gravity.CENTER_HORIZONTAL; return this }

/** Beeps and vibration, so trials work with the screen facing away from the tester. */
internal class Feedback(private val activity: Activity) {
    private val tone: ToneGenerator? = try { ToneGenerator(android.media.AudioManager.STREAM_MUSIC, 90) } catch (e: RuntimeException) { null }

    fun beep(type: Int = ToneGenerator.TONE_PROP_BEEP, ms: Int = 120) {
        try { tone?.startTone(type, ms) } catch (e: RuntimeException) { /* audio unavailable */ }
    }

    fun vibrate(ms: Long) {
        val v = activity.getSystemService(Vibrator::class.java) ?: return
        if (v.hasVibrator()) v.vibrate(VibrationEffect.createOneShot(ms, VibrationEffect.DEFAULT_AMPLITUDE))
    }

    fun release() { tone?.release() }
}
