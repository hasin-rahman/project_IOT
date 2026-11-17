package com.posefl.app

import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

class ServerException(val code: Int, message: String) : IOException(message)

/** Minimal JSON-over-HTTP client for the FL server. Call from a background thread. */
class ServerClient(baseUrl: String) {
    private val base = baseUrl.trim().trimEnd('/')

    init {
        require(base.startsWith("http://") || base.startsWith("https://")) {
            "Server URL must start with http://, e.g. http://192.168.1.10:8000"
        }
    }

    fun get(path: String): JSONObject = request("GET", path, null)

    fun post(path: String, body: JSONObject): JSONObject = request("POST", path, body)

    private fun request(method: String, path: String, body: JSONObject?): JSONObject {
        val conn = URL(base + path).openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = 5_000
            conn.readTimeout = 20_000
            conn.setRequestProperty("Content-Type", "application/json")
            if (body != null) {
                val bytes = body.toString().toByteArray(Charsets.UTF_8)
                conn.doOutput = true
                conn.setFixedLengthStreamingMode(bytes.size)
                conn.outputStream.use { it.write(bytes) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
            val json = if (text.isNotBlank()) JSONObject(text) else JSONObject()
            if (code !in 200..299) throw ServerException(code, json.optString("error", "HTTP $code"))
            return json
        } finally {
            conn.disconnect()
        }
    }
}
