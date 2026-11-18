package com.posefl.core

import java.math.BigInteger
import java.nio.ByteBuffer
import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.Mac
import javax.crypto.spec.SecretKeySpec

/**
 * Secure aggregation for two clients (our extra-credit extension).
 * Goal: the server can compute the AVERAGE of the phones' models without ever seeing one phone's model.
 *
 *  1. Key agreement (Diffie-Hellman). Each phone picks a secret number x and publishes g^x mod p.
 *     The server only relays these public values. Each phone combines its own secret with the other
 *     phone's public value and both get the same number g^(xy) mod p, which the server cannot compute.
 *     The shared key is the SHA-256 hash of that number.
 *  2. Masking. Every round, both phones turn the shared key into the same list of random 64-bit
 *     numbers (the mask). Each phone converts  numSamples * weight  to a whole number; phone A ADDS
 *     the mask and phone B SUBTRACTS it, modulo 2^64. What is uploaded looks like random numbers.
 *  3. Unmasking by adding. The server adds the two uploads modulo 2^64: the masks cancel and
 *     n_A*w_A + n_B*w_B is left. Divided by n_A + n_B that is exactly the FedAvg result.
 *
 * The same shared key also encrypts the local-only models that the server relays between the phones.
 * The server side (and a line-by-line Python copy of this file) is server/secure_agg.py.
 *
 * Limits: with two clients each phone can work out the other's update from the average, so the
 * protection is against the server. The server is assumed to relay the public keys honestly.
 */
object SecureAggregation {
    /** 2048-bit Diffie-Hellman group 14 from RFC 3526 (a standard, public prime) and generator 2. */
    val DH_PRIME = BigInteger(
        "FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DD" +
            "EF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7ED" +
            "EE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F" +
            "83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3B" +
            "E39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA0510" +
            "15728E5A8AACAA68FFFFFFFFFFFFFFFF", 16)
    val DH_GENERATOR: BigInteger = BigInteger.valueOf(2)

    /** Weights are sent as whole numbers: round(numSamples * weight * SCALE). */
    const val SCALE = 16777216.0   // 2^24

    // ------------------------------------------------------------------ key agreement

    /** A fresh 256-bit secret number. It never leaves the phone. */
    fun newPrivateKey(): BigInteger = BigInteger(256, SecureRandom()).setBit(255)

    /** What a phone publishes: g^x mod p, as a hex string. */
    fun publicKey(privateKey: BigInteger): String = DH_GENERATOR.modPow(privateKey, DH_PRIME).toString(16)

    /** The 32-byte key both phones end up with: SHA-256 of g^(xy) mod p. */
    fun sharedKey(privateKey: BigInteger, otherPublicHex: String): ByteArray {
        val other = BigInteger(otherPublicHex, 16)
        require(other > BigInteger.ONE && other < DH_PRIME - BigInteger.ONE) { "invalid public key" }
        return sha256(other.modPow(privateKey, DH_PRIME).toString(16).toByteArray(Charsets.UTF_8))
    }

    // ------------------------------------------------------------------ random numbers from the shared key

    /** [count] pseudo-random 64-bit numbers. Both phones get the same list for the same key and label. */
    fun randomWords(key: ByteArray, label: String, count: Int): LongArray {
        val words = LongArray(count)
        var filled = 0
        var block = 0
        while (filled < count) {
            val digest = ByteBuffer.wrap(sha256(key + "$label:$block".toByteArray(Charsets.UTF_8)))
            repeat(4) { if (filled < count) words[filled++] = digest.getLong() else digest.getLong() }
            block++
        }
        return words
    }

    /** A different mask for every round (and every retry of a round), so a mask is never used twice. */
    fun maskLabel(runId: String, round: Int, attempt: Int) = "mask|$runId|$round|$attempt"

    // ------------------------------------------------------------------ masking

    /**
     * [weights] are all model parameters in one array (W then b). Returns the masked values as
     * 16-digit hex strings. Kotlin's Long arithmetic wraps around, which is exactly "modulo 2^64".
     */
    fun maskUpdate(weights: DoubleArray, numSamples: Int, key: ByteArray, label: String, addsMask: Boolean): List<String> {
        val mask = randomWords(key, label, weights.size)
        return weights.indices.map { i ->
            val wholeNumber = Math.round(numSamples * weights[i] * SCALE)
            val masked = if (addsMask) wholeNumber + mask[i] else wholeNumber - mask[i]
            String.format("%016x", masked)
        }
    }

    /** One phone adds the mask and the other subtracts it; the smaller id adds. */
    fun addsMask(clientId: String, otherId: String): Boolean = clientId < otherId

    /** What ONE masked upload looks like if someone tries to read it as weights (it is noise). */
    fun decodeAlone(masked: List<String>, numSamples: Int): DoubleArray =
        DoubleArray(masked.size) { java.lang.Long.parseUnsignedLong(masked[it], 16) / (SCALE * numSamples) }

    // ------------------------------------------------------------------ compressed secure aggregation
    // Shiv's 8-bit compression combined with the masking above (server/secure_agg.py has the same code).
    //
    // Instead of the full weights, the phone sends the CHANGE it made this round (delta = new - global),
    // multiplied by its share of all training data (weight = n / total), clipped to +-clip and rounded to
    // one of 255 levels (8 bits). The server announces `clip`, so both phones use the same scale. The two
    // weighted values add up to at most +-127 levels, so a 16-bit mask is enough: 2 bytes per parameter.

    const val LEVELS = 127

    /** Its own label, so the 16-bit masks never repeat the 64-bit ones. */
    fun compressedLabel(runId: String, round: Int, attempt: Int) = "cmask|$runId|$round|$attempt"

    /** The upload: base64 of the 16-bit masked numbers, and how many values had to be clipped. */
    class Compressed(val payload: String, val clipped: Int)

    /** Quantize the weighted change to 8 bits, add (or subtract) a 16-bit mask, and pack it as 2 bytes per parameter. */
    fun maskCompressed(delta: DoubleArray, weight: Double, clip: Double, key: ByteArray, label: String, addsMask: Boolean): Compressed {
        val mask = randomWords(key, label, delta.size)
        val packed = ByteBuffer.allocate(2 * delta.size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        var clipped = 0
        for (i in delta.indices) {
            var d = delta[i]
            if (kotlin.math.abs(d) > clip) { clipped++; d = if (d > 0) clip else -clip }
            val level = Math.round(weight * d / clip * LEVELS).toInt()        // a whole number from -127 to 127
            val m = (mask[i] and 0xFFFF).toInt()                               // a 16-bit mask
            val masked = (if (addsMask) level + m else level - m) and 0xFFFF   // modulo 2^16
            packed.putShort(masked.toShort())
        }
        return Compressed(java.util.Base64.getEncoder().encodeToString(packed.array()), clipped)
    }

    /** What ONE compressed masked upload looks like if someone reads it as a change of the weights (noise). */
    fun decodeCompressedAlone(payload: String, weight: Double, clip: Double): DoubleArray {
        val bytes = ByteBuffer.wrap(java.util.Base64.getDecoder().decode(payload)).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        return DoubleArray(bytes.remaining() / 2) { bytes.getShort().toDouble() * clip / LEVELS / weight }
    }

    // ------------------------------------------------------------------ encrypting the relayed local models

    /**
     * XORs the text with random bytes made from the shared key, and adds a check value (HMAC-SHA256)
     * so the receiver can tell whether it has the right key.
     */
    fun encrypt(key: ByteArray, label: String, text: String): Map<String, Any?> {
        val data = text.toByteArray(Charsets.UTF_8)
        val ciphertext = xorWithKeyStream(key, label, data)
        return linkedMapOf("ciphertext" to hex(ciphertext), "tag" to hex(checkValue(key, label, ciphertext)))
    }

    /** Returns the text, or null if the check value does not match (wrong or outdated key). */
    fun decrypt(key: ByteArray, label: String, blob: Map<String, Any?>): String? {
        val ciphertext = unhex(blob.str("ciphertext"))
        if (hex(checkValue(key, label, ciphertext)) != blob.str("tag")) return null
        return String(xorWithKeyStream(key, label, ciphertext), Charsets.UTF_8)
    }

    private fun xorWithKeyStream(key: ByteArray, label: String, data: ByteArray): ByteArray {
        val words = randomWords(key, "enc|$label", (data.size + 7) / 8)
        val stream = ByteBuffer.allocate(words.size * 8)
        for (w in words) stream.putLong(w)
        val bytes = stream.array()
        return ByteArray(data.size) { (data[it].toInt() xor bytes[it].toInt()).toByte() }
    }

    private fun checkValue(key: ByteArray, label: String, ciphertext: ByteArray): ByteArray {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(key, "HmacSHA256"))
        mac.update("tag|$label".toByteArray(Charsets.UTF_8))
        return mac.doFinal(ciphertext)
    }

    private fun sha256(data: ByteArray): ByteArray = MessageDigest.getInstance("SHA-256").digest(data)

    private fun hex(bytes: ByteArray): String = bytes.joinToString("") { String.format("%02x", it.toInt() and 0xff) }

    private fun unhex(text: String): ByteArray = ByteArray(text.length / 2) { text.substring(2 * it, 2 * it + 2).toInt(16).toByte() }
}

/**
 * One phone's secrets for one training run: its private key and, once the other phone's public
 * key has arrived, the shared key.
 */
class SecureSession(private val clientId: String, private val otherId: String, private val runId: String) {
    private val privateKey = SecureAggregation.newPrivateKey()
    /** Sent to the server, which passes it on to the other phone. */
    val publicKey: String = SecureAggregation.publicKey(privateKey)
    private var sharedKey: ByteArray? = null
    /** The server's key counter when the shared key was computed (it changes if a phone restarts). */
    var keyEpoch = -1
        private set

    val ready: Boolean get() = sharedKey != null

    fun agree(otherPublicKey: String, epoch: Int) {
        sharedKey = SecureAggregation.sharedKey(privateKey, otherPublicKey)
        keyEpoch = epoch
    }

    /** The masked version of [model] that is uploaded instead of the real weights. */
    fun mask(model: SoftmaxModel, numSamples: Int, round: Int, attempt: Int): List<String> =
        SecureAggregation.maskUpdate(model.W + model.b, numSamples, sharedKey!!,
            SecureAggregation.maskLabel(runId, round, attempt), SecureAggregation.addsMask(clientId, otherId))

    /** Compressed secure aggregation: the 8-bit, 16-bit-masked version of this round's change of the weights. */
    fun maskCompressed(delta: DoubleArray, weight: Double, clip: Double, round: Int, attempt: Int): SecureAggregation.Compressed =
        SecureAggregation.maskCompressed(delta, weight, clip, sharedKey!!,
            SecureAggregation.compressedLabel(runId, round, attempt), SecureAggregation.addsMask(clientId, otherId))

    /** Encrypts this phone's local-only model so that only the other phone can read it. */
    fun encryptLocalModel(modelJson: String): Map<String, Any?> =
        SecureAggregation.encrypt(sharedKey!!, "local|$runId|$clientId", modelJson)

    /** Opens the other phone's local-only model, or null if it was encrypted with an outdated key. */
    fun decryptOtherLocalModel(blob: Map<String, Any?>): String? =
        sharedKey?.let { SecureAggregation.decrypt(it, "local|$runId|$otherId", blob) }
}
