package com.fedpose.app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager

/**
 * Registers the four sensors at ~50 Hz and forwards every event as (name, timestampNs, x, y, z).
 * Names match the raw CSV format: acc, grav, gyro, rot. [sink] can be swapped at any time
 * (recording a session, or feeding the live predictor).
 */
class SensorRecorder(context: Context) : SensorEventListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    private val types = mapOf(
        Sensor.TYPE_ACCELEROMETER to "acc",
        Sensor.TYPE_GRAVITY to "grav",
        Sensor.TYPE_GYROSCOPE to "gyro",
        Sensor.TYPE_ROTATION_VECTOR to "rot",
    )

    @Volatile var sink: ((String, Long, Double, Double, Double) -> Unit)? = null

    /** Sensors this phone does not have (the feature set must avoid them). */
    fun missing(): List<String> = types.filter { sm.getDefaultSensor(it.key) == null }.values.toList()

    fun start() {
        for (type in types.keys) sm.getDefaultSensor(type)?.let { sm.registerListener(this, it, 20_000) }  // 20 ms = 50 Hz
    }

    fun stop() = sm.unregisterListener(this)

    override fun onSensorChanged(e: SensorEvent) {
        val name = types[e.sensor.type] ?: return
        sink?.invoke(name, e.timestamp, e.values[0].toDouble(), e.values[1].toDouble(), e.values[2].toDouble())
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
}
