package com.posefl.app

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Handler
import android.os.HandlerThread
import kotlin.math.max
import kotlin.math.sqrt

/**
 * Listens to the accelerometer, gravity, gyroscope and rotation-vector sensors at ~50 Hz.
 *
 * The sensors fire independently, so we keep the latest value of each and write one time-aligned
 * row per accelerometer event (at most one row every [MIN_ROW_GAP_NS], i.e. ~50 Hz even on phones
 * that deliver faster). Row layout = SensorCsv.HEADER:
 *   t_ms, ax, ay, az, gx, gy, gz, wx, wy, wz, rx, ry, rz, rw
 */
class SensorRecorder(context: Context) : SensorEventListener {
    private val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    val accelSensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    val gravitySensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GRAVITY)
    val gyroSensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    val rotationSensor: Sensor? = sm.getDefaultSensor(Sensor.TYPE_ROTATION_VECTOR)

    private val lock = Any()
    private val acc = DoubleArray(3)
    private val grav = DoubleArray(3)
    private val gyro = DoubleArray(3)
    private val quat = doubleArrayOf(0.0, 0.0, 0.0, 1.0)
    private var lowPassReady = false

    private var rows = ArrayList<DoubleArray>()
    private val live = ArrayDeque<DoubleArray>()   // last ~4 s for live prediction, t = ms since boot
    private var lastLiveNs = -1L
    private var recording = false
    private var startNs = -1L
    private var lastRowNs = -1L
    private var thread: HandlerThread? = null

    /** Comma-separated list of available sensors, stored with every session. */
    fun availableSensors(): String = listOfNotNull(
        accelSensor?.let { "accelerometer" },
        gravitySensor?.let { "gravity" } ?: "gravity(low-pass fallback)",
        gyroSensor?.let { "gyroscope" },
        rotationSensor?.let { "rotation_vector" },
    ).joinToString(",")

    fun start() {
        if (thread != null) return
        val t = HandlerThread("sensors").also { it.start() }
        thread = t
        val handler = Handler(t.looper)
        for (s in listOfNotNull(accelSensor, gravitySensor, gyroSensor, rotationSensor)) {
            sm.registerListener(this, s, SAMPLING_PERIOD_US, handler)
        }
    }

    fun stop() {
        sm.unregisterListener(this)
        thread?.quitSafely()
        thread = null
    }

    fun beginRecording() = synchronized(lock) {
        rows = ArrayList(4000)
        recording = true
        startNs = -1L
        lastRowNs = -1L
    }

    fun endRecording(): List<DoubleArray> = synchronized(lock) {
        recording = false
        rows
    }

    fun rowCount(): Int = synchronized(lock) { rows.size }

    fun latestGravity(): DoubleArray = synchronized(lock) { grav.copyOf() }

    /** The most recent rows (same layout as recordings, oldest first) for the live demo screen. */
    fun liveRows(): List<DoubleArray> = synchronized(lock) { ArrayList(live) }

    override fun onSensorChanged(e: SensorEvent) {
        synchronized(lock) {
            val v = e.values
            when (e.sensor.type) {
                Sensor.TYPE_ACCELEROMETER -> {
                    for (i in 0..2) acc[i] = v[i].toDouble()
                    if (gravitySensor == null) { // fallback: low-pass filter the accelerometer
                        for (i in 0..2) grav[i] = if (lowPassReady) 0.9 * grav[i] + 0.1 * acc[i] else acc[i]
                        lowPassReady = true
                    }
                    if (lastLiveNs < 0 || e.timestamp - lastLiveNs >= MIN_ROW_GAP_NS) {
                        lastLiveNs = e.timestamp
                        live.addLast(doubleArrayOf(
                            e.timestamp / 1e6,
                            acc[0], acc[1], acc[2],
                            grav[0], grav[1], grav[2],
                            gyro[0], gyro[1], gyro[2],
                            quat[0], quat[1], quat[2], quat[3],
                        ))
                        while (live.size > LIVE_ROWS) live.removeFirst()
                    }
                    if (recording) {
                        if (startNs < 0) startNs = e.timestamp
                        if (lastRowNs < 0 || e.timestamp - lastRowNs >= MIN_ROW_GAP_NS) {
                            lastRowNs = e.timestamp
                            rows.add(doubleArrayOf(
                                (e.timestamp - startNs) / 1e6,
                                acc[0], acc[1], acc[2],
                                grav[0], grav[1], grav[2],
                                gyro[0], gyro[1], gyro[2],
                                quat[0], quat[1], quat[2], quat[3],
                            ))
                        }
                    }
                }
                Sensor.TYPE_GRAVITY -> for (i in 0..2) grav[i] = v[i].toDouble()
                Sensor.TYPE_GYROSCOPE -> for (i in 0..2) gyro[i] = v[i].toDouble()
                Sensor.TYPE_ROTATION_VECTOR -> {
                    val x = v[0].toDouble(); val y = v[1].toDouble(); val z = v[2].toDouble()
                    quat[0] = x; quat[1] = y; quat[2] = z
                    quat[3] = if (v.size >= 4) v[3].toDouble() else sqrt(max(0.0, 1 - x * x - y * y - z * z))
                }
            }
        }
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}

    companion object {
        const val SAMPLING_PERIOD_US = 20_000          // request 50 Hz
        const val MIN_ROW_GAP_NS = 15_000_000L         // keep <= ~50 rows/s even if delivered faster
        const val LIVE_ROWS = 250                      // ~4-5 s of live history
    }
}
