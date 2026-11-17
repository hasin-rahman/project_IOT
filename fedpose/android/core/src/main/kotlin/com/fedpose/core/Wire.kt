package com.fedpose.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Base64
import kotlin.math.abs
import kotlin.math.sqrt

const val CODEC_F32 = "f32"
const val CODEC_INT8 = "int8"

/**
 * Client <-> server wire helpers. Twin of python/fedpose/wire.py.
 *   f32  : float32 little endian, 4 bytes per parameter
 *   int8 : symmetric int8 + one float32 scale; scale = max|d| / 127 (as float32), q = rint(d / scale)
 */
object Wire {
    private const val QMAX = 127

    fun f32Round(w: DoubleArray): DoubleArray = DoubleArray(w.size) { w[it].toFloat().toDouble() }

    fun f32ToB64(d: DoubleArray): String {
        val bb = ByteBuffer.allocate(4 * d.size).order(ByteOrder.LITTLE_ENDIAN)
        for (v in d) bb.putFloat(v.toFloat())
        return Base64.getEncoder().encodeToString(bb.array())
    }

    fun b64ToF32(s: String): DoubleArray {
        val bb = ByteBuffer.wrap(Base64.getDecoder().decode(s)).order(ByteOrder.LITTLE_ENDIAN)
        return DoubleArray(bb.remaining() / 4) { bb.float.toDouble() }
    }

    /** Returns (scale, base64 payload). */
    fun encodeInt8(delta: DoubleArray): Pair<Double, String> {
        var maxAbs = 0.0
        for (v in delta) if (abs(v) > maxAbs) maxAbs = abs(v)
        val scale: Double = if (maxAbs > 0) (maxAbs / QMAX).toFloat().toDouble() else 1.0
        val q = ByteArray(delta.size) {
            Math.rint(delta[it] / scale).coerceIn(-QMAX.toDouble(), QMAX.toDouble()).toInt().toByte()
        }
        return scale to Base64.getEncoder().encodeToString(q)
    }

    fun decodeInt8(scale: Double, b64: String): DoubleArray {
        val q = Base64.getDecoder().decode(b64)
        val sc = scale.toFloat().toDouble()
        return DoubleArray(q.size) { q[it].toDouble() * sc }
    }

    fun encodeDelta(delta: DoubleArray, codec: String): Map<String, Any?> = when (codec) {
        CODEC_F32 -> mapOf("codec" to CODEC_F32, "payload" to f32ToB64(delta))
        CODEC_INT8 -> encodeInt8(delta).let { (s, p) -> mapOf("codec" to CODEC_INT8, "scale" to s, "payload" to p) }
        else -> error("unknown codec $codec")
    }

    fun updateBody(clientId: String, round: Int, baseVersion: Int, nSamples: Int, delta: DoubleArray,
                   codec: String, extra: Map<String, Any?> = emptyMap()): Map<String, Any?> {
        val m = LinkedHashMap<String, Any?>()
        m["client_id"] = clientId
        m["round"] = round
        m["base_version"] = baseVersion
        m["n_samples"] = nSamples
        m["skipped"] = false
        m.putAll(encodeDelta(delta, codec))
        m.putAll(extra)
        return m
    }

    fun skipBody(clientId: String, round: Int, baseVersion: Int, updateNorm: Double): Map<String, Any?> =
        linkedMapOf(
            "client_id" to clientId, "round" to round, "base_version" to baseVersion,
            "n_samples" to 0, "skipped" to true, "update_norm" to updateNorm,
        )

    fun l2(d: DoubleArray): Double { var s = 0.0; for (v in d) s += v * v; return sqrt(s) }
}
