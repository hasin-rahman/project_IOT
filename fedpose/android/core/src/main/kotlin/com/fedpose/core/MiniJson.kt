package com.fedpose.core

/**
 * Tiny dependency free JSON reader/writer (org.json does not exist on a plain JVM, and we want
 * this module to be unit-testable off device). Supports objects, arrays, strings, numbers,
 * booleans and null. Integers parse to Long, everything else numeric to Double.
 */
object MiniJson {
    fun parse(text: String): Any? {
        val p = Parser(text)
        p.ws()
        val v = p.value()
        p.ws()
        require(p.i == text.length) { "trailing characters at ${p.i}" }
        return v
    }

    @Suppress("UNCHECKED_CAST")
    fun parseObject(text: String): Map<String, Any?> = parse(text) as Map<String, Any?>

    fun write(v: Any?): String = StringBuilder().also { w(it, v) }.toString()

    private fun w(sb: StringBuilder, v: Any?) {
        when (v) {
            null -> sb.append("null")
            is Boolean -> sb.append(if (v) "true" else "false")
            is Int, is Long -> sb.append(v.toString())
            is Float -> { require(v.isFinite()); sb.append(v.toDouble().toString()) }
            is Double -> { require(v.isFinite()) { "non finite number" }; sb.append(v.toString()) }
            is String -> str(sb, v)
            is Map<*, *> -> {
                sb.append('{')
                var first = true
                for ((k, x) in v) {
                    if (!first) sb.append(',')
                    first = false
                    str(sb, k as String); sb.append(':'); w(sb, x)
                }
                sb.append('}')
            }
            is List<*> -> {
                sb.append('[')
                v.forEachIndexed { i, x -> if (i > 0) sb.append(','); w(sb, x) }
                sb.append(']')
            }
            is DoubleArray -> w(sb, v.toList())
            is IntArray -> w(sb, v.toList())
            else -> error("unsupported json type ${v::class}")
        }
    }

    private fun str(sb: StringBuilder, s: String) {
        sb.append('"')
        for (c in s) when {
            c == '"' -> sb.append("\\\"")
            c == '\\' -> sb.append("\\\\")
            c == '\n' -> sb.append("\\n")
            c == '\r' -> sb.append("\\r")
            c == '\t' -> sb.append("\\t")
            c < ' ' -> sb.append(String.format("\\u%04x", c.code))
            else -> sb.append(c)
        }
        sb.append('"')
    }

    private class Parser(val s: String) {
        var i = 0
        fun ws() { while (i < s.length && s[i].isWhitespace()) i++ }
        fun value(): Any? {
            ws()
            require(i < s.length) { "unexpected end" }
            return when (val c = s[i]) {
                '{' -> obj()
                '[' -> arr()
                '"' -> string()
                't' -> lit("true", true)
                'f' -> lit("false", false)
                'n' -> lit("null", null)
                else -> if (c == '-' || c.isDigit()) number() else error("bad char '$c' at $i")
            }
        }
        fun lit(t: String, v: Any?): Any? {
            require(s.startsWith(t, i)) { "bad literal at $i" }
            i += t.length
            return v
        }
        fun obj(): Map<String, Any?> {
            val m = LinkedHashMap<String, Any?>()
            i++; ws()
            if (s[i] == '}') { i++; return m }
            while (true) {
                ws(); val k = string(); ws()
                require(s[i] == ':') { "expected ':' at $i" }; i++
                m[k] = value(); ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == '}') { "expected '}' at $i" }; i++
                return m
            }
        }
        fun arr(): List<Any?> {
            val l = ArrayList<Any?>()
            i++; ws()
            if (s[i] == ']') { i++; return l }
            while (true) {
                l.add(value()); ws()
                if (s[i] == ',') { i++; continue }
                require(s[i] == ']') { "expected ']' at $i" }; i++
                return l
            }
        }
        fun string(): String {
            require(s[i] == '"') { "expected string at $i" }
            i++
            val sb = StringBuilder()
            while (s[i] != '"') {
                var c = s[i++]
                if (c == '\\') {
                    c = s[i++]
                    when (c) {
                        'n' -> sb.append('\n'); 't' -> sb.append('\t'); 'r' -> sb.append('\r')
                        'b' -> sb.append('\b'); 'f' -> sb.append('\u000C')
                        'u' -> { sb.append(s.substring(i, i + 4).toInt(16).toChar()); i += 4 }
                        else -> sb.append(c)
                    }
                } else sb.append(c)
            }
            i++
            return sb.toString()
        }
        fun number(): Any {
            val st = i
            if (s[i] == '-') i++
            while (i < s.length && (s[i].isDigit() || s[i] in ".eE+-")) i++
            val t = s.substring(st, i)
            return if (t.any { it == '.' || it == 'e' || it == 'E' }) t.toDouble() else t.toLong()
        }
    }
}

fun Map<String, Any?>.int(k: String): Int = (this[k] as Number).toInt()
fun Map<String, Any?>.dbl(k: String): Double = (this[k] as Number).toDouble()
fun Map<String, Any?>.str(k: String): String = this[k] as String
fun Map<String, Any?>.bool(k: String): Boolean = this[k] as? Boolean ?: false
