package com.posefl.app

import android.content.Context

/** Per-phone settings: which FL client this phone is, who is recording, and where the server is. */
class AppPrefs(context: Context) {
    private val sp = context.getSharedPreferences("posefl", Context.MODE_PRIVATE)

    var clientId: String
        get() = sp.getString("client_id", "A") ?: "A"
        set(v) = sp.edit().putString("client_id", v).apply()

    var participant: String
        get() = sp.getString("participant", "") ?: ""
        set(v) = sp.edit().putString("participant", sanitize(v)).apply()

    var serverUrl: String
        get() = sp.getString("server_url", "http://192.168.1.10:8000") ?: ""
        set(v) = sp.edit().putString("server_url", v.trim().trimEnd('/')).apply()

    /** Training screen: keep training whenever the server starts a new run (sweeps). */
    var followRuns: Boolean
        get() = sp.getBoolean("follow_runs", false)
        set(v) = sp.edit().putBoolean("follow_runs", v).apply()

    companion object {
        /** Participant IDs end up in file names, so keep them to letters, digits, '-' and '_'. */
        fun sanitize(s: String): String = s.trim().replace(Regex("[^A-Za-z0-9_-]"), "_").take(32)
    }
}
