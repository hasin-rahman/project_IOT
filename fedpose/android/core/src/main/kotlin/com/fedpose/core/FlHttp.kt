package com.fedpose.core

import java.net.HttpURLConnection

/** Counts HTTP BODY bytes only (headers excluded), per client instance. */
class TrafficCounter {
    @Volatile var bytesSent = 0L
        private set
    @Volatile var bytesReceived = 0L
        private set
    @Volatile var requests = 0
        private set

    @Synchronized fun add(sent: Int, received: Int) {
        bytesSent += sent; bytesReceived += received; requests++
    }
}

class HttpResult(val code: Int, val body: Map<String, Any?>)

/** Minimal JSON-over-HTTP client using only java.net (works on a JVM and on Android). */
class FlHttp(
    private val baseUrl: String,
    private val connectTimeoutMs: Int = 3000,
    private val readTimeoutMs: Int = 10000,
) {
    val traffic = TrafficCounter()

    fun get(path: String): HttpResult = request("GET", path, null)
    fun post(path: String, body: Map<String, Any?>): HttpResult =
        request("POST", path, MiniJson.write(body).toByteArray(Charsets.UTF_8))

    /** Throws java.io.IOException when the server cannot be reached (callers retry). */
    private fun request(method: String, path: String, body: ByteArray?): HttpResult {
        val c = java.net.URI(baseUrl.trimEnd('/') + path).toURL().openConnection() as HttpURLConnection
        try {
            c.requestMethod = method
            c.connectTimeout = connectTimeoutMs
            c.readTimeout = readTimeoutMs
            if (body != null) {
                c.doOutput = true
                c.setRequestProperty("Content-Type", "application/json")
                c.setFixedLengthStreamingMode(body.size)
                c.outputStream.use { it.write(body) }
            }
            val code = c.responseCode
            val stream = if (code >= 400) c.errorStream else c.inputStream
            val bytes = stream?.use { it.readBytes() } ?: ByteArray(0)
            traffic.add(body?.size ?: 0, bytes.size)
            val parsed = if (bytes.isEmpty()) emptyMap() else MiniJson.parseObject(String(bytes, Charsets.UTF_8))
            return HttpResult(code, parsed)
        } finally {
            c.disconnect()
        }
    }
}
