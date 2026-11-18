package com.posefl.core

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.abs

/** JVM unit tests for the pure-Kotlin core (run from Android Studio: right-click > Run 'CoreTest'). */
class CoreTest {

    /** A synthetic 6-second still session at 50 Hz with gravity along the pose's axis. */
    private fun session(pose: Pose, seconds: Int = 6): List<DoubleArray> = (0 until seconds * 50).map { i ->
        val g = DoubleArray(3).also { it[pose.axis] = 9.81 * pose.sign }
        doubleArrayOf(i * 20.0, g[0], g[1], g[2], g[0], g[1], g[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
    }

    @Test
    fun windowsAreTrimmedAndOverlapping() {
        // samples at t = 0..5980 ms; trimming 1 s each end leaves [1000, 4980] ms.
        // 1 s windows every 0.5 s must end by 4980 -> starts 1000, 1500, ..., 3500 = 6 windows
        assertEquals(6, FeatureExtractor.windowRanges(session(Pose.SCREEN_UP)).size)
    }

    @Test
    fun featureSizesMatchSpec() {
        assertEquals(12, FeatureSets.size("acc+grav"))
        assertEquals(24, FeatureSets.size("acc+grav+gyro+rot"))
        val f = FeatureExtractor.extract(session(Pose.PORTRAIT_UP), "acc+grav")
        assertEquals(12, f[0].size)
        assertEquals(1.0, f[0][1], 1e-9)  // acc mean y = +1 g
    }

    @Test
    fun rotationUpVectorForFlatPhoneIsPlusZ() {
        val u = FeatureExtractor.upFromQuaternion(0.0, 0.0, 0.0, 1.0)
        assertEquals(0.0, u[0], 1e-12); assertEquals(0.0, u[1], 1e-12); assertEquals(1.0, u[2], 1e-12)
    }

    @Test
    fun ruleBasedGuessMatchesEveryPose() {
        for (p in Pose.entries) {
            val g = DoubleArray(3).also { it[p.axis] = 9.81 * p.sign }
            assertEquals(p, Pose.guessFromGravity(g[0], g[1], g[2]))
        }
    }

    @Test
    fun modelLearnsAllSixPoses() {
        val X = ArrayList<DoubleArray>()
        val y = ArrayList<Int>()
        for (p in Pose.entries) {
            val f = FeatureExtractor.extract(session(p), "acc+grav")
            X.addAll(f); repeat(f.size) { y.add(p.index) }
        }
        val m = SoftmaxModel(12)
        assertEquals(78, m.numParams)
        m.train(X, y.toIntArray(), epochs = 20, rng = java.util.Random(0))
        val pred = IntArray(X.size) { m.predict(X[it]) }
        assertEquals(1.0, Metrics(y.toIntArray(), pred).accuracy, 1e-12)
    }

    @Test
    fun fedAvgIsSampleWeighted() {
        val a = SoftmaxModel(2, 2).also { it.W.fill(1.0) }
        val b = SoftmaxModel(2, 2).also { it.W.fill(5.0) }
        val avg = SoftmaxModel.average(listOf(a, b), listOf(10, 30))
        assertTrue(avg.W.all { abs(it - 4.0) < 1e-12 })
    }

    // ---------------- federated client support (core/Json.kt, core/FlClient.kt, core/Live.kt) ----------------

    @Test
    fun jsonRoundTrip() {
        val text = Json.stringify(mapOf("a" to 1, "b" to doubleArrayOf(0.5, -2.0), "s" to "x\"y", "n" to null, "t" to true))
        val back = Json.parseObject(text)
        assertEquals(1, back.int("a"))
        assertEquals(-2.0, back.doubles("b")[1], 0.0)
        assertEquals("x\"y", back.str("s"))
        assertTrue(back.containsKey("n") && back["n"] == null)
        assertTrue(back.bool("t"))
    }

    @Test
    fun modelSurvivesWireFormat() {
        val m = SoftmaxModel(12).also { for (i in it.W.indices) it.W[i] = i * 0.01 - 0.3 }
        val back = ModelIo.fromMap(Json.parseObject(Json.stringify(ModelIo.toMap(m, "acc+grav"))))
        assertTrue(m.W.indices.all { m.W[it] == back.W[it] })
    }

    private fun labeled(pose: Pose, split: String, who: String) = LabeledSession("$who-${pose.key}-$split", pose, split, who, session(pose))

    @Test
    fun localDataKeepsTheNonIidSplit() {
        val sessions = Pose.entries.flatMap { p -> listOf(labeled(p, "train", "ann"), labeled(p, "val", "ann"), labeled(p, "test", "ann"), labeled(p, "test", "zed")) }
        val d = LocalData.build(sessions, "A", "acc+grav")
        val trainClasses = d.train.y.toSet()
        assertEquals(setOf(0, 1, 2), trainClasses)            // only client A's poses, even if B poses were labelled train
        assertEquals(6, d.validation.y.toSet().size)
        assertEquals(d.test.size / 2, d.testUnseen.size)       // "zed" gave no training data
    }

    @Test
    fun livePredictorAndTestMode() {
        val sessions = Pose.entries.map { labeled(it, "train", "ann") }
        val data = LocalData.build(sessions, "all", "acc+grav")
        val m = SoftmaxModel(12).also { it.train(data.train.X, data.train.y, epochs = 30, rng = java.util.Random(1)) }
        val models = listOf(StoredModel("global", m, "acc+grav", "run", 0L))
        val predictor = LivePredictor()
        val test = LiveTest(listOf("global"), seed = 3)
        repeat(6) {
            val target = test.nextTarget()
            predictor.reset()
            val frame = predictor.update(session(target, seconds = 2), models)
            assertTrue(!frame.moving && !frame.warmingUp)
            test.record(target, frame.predictions.associate { it.modelName to it.best }, "ann")
        }
        assertEquals(6, test.trials.map { it.target }.toSet().size)   // every pose asked once per block
        assertEquals(1.0, test.accuracy("global"), 1e-12)
        assertTrue(test.toCsv().lines().size >= 7)
    }

    // ---------------- demo-day pieces: independent baselines, rubric score, blind-test log ----------------

    @Test
    fun localOnlyModelFailsOnTheOtherClientsPoses() {
        val sessions = Pose.entries.flatMap { p -> listOf(labeled(p, "train", "ann"), labeled(p, "test", "ann")) }
        val d = LocalData.build(sessions, "A", "acc+grav")
        assertEquals(listOf(4, 4, 4, 0, 0, 0), d.train.classCounts().toList().map { if (it > 0) 4 else 0 })
        val localA = SoftmaxModel(12).also { it.train(d.train.X, d.train.y, epochs = 20, rng = java.util.Random(0)) }
        val m = localA.metricsOn(d.test)
        assertEquals(1.0, m.accuracyOn(Pose.trainPosesFor("A"))!!, 1e-12)   // its own three poses
        assertEquals(0.0, m.accuracyOn(Pose.trainPosesFor("B"))!!, 1e-12)   // never seen in training
    }

    @Test
    fun practiceScoreFollowsTheRubric() {
        val test = LiveTest(ModelNames.ALL, seed = 5)
        // global always right; each local model right on its own poses, and answering one of its own poses otherwise
        repeat(36) {
            val target = test.nextTarget()
            val a = if (target in Pose.trainPosesFor("A")) target else Pose.SCREEN_UP
            val b = if (target in Pose.trainPosesFor("B")) target else Pose.PORTRAIT_DOWN
            test.record(target, mapOf(ModelNames.GLOBAL to target, ModelNames.LOCAL_A to a, ModelNames.LOCAL_B to b), "ann")
        }
        for (p in Pose.entries) assertEquals(6, test.countOn(listOf(p)))        // every pose 6 times in 36 trials
        assertEquals(18, test.correctOn(ModelNames.LOCAL_A, Pose.trainPosesFor("A")))
        assertEquals(0, test.correctOn(ModelNames.LOCAL_A, Pose.trainPosesFor("B")))
        assertEquals(40.0, test.rubricPoints(), 1e-9)
        test.undoLast()
        assertEquals(35, test.trials.size)
    }

    @Test
    fun blindTestLogNumbersPlacementsAndUsesOneWindow() {
        val log = BlindTestLog()
        val answers = mapOf(ModelNames.GLOBAL to Pose.LANDSCAPE_LEFT, ModelNames.LOCAL_A to Pose.PORTRAIT_UP, ModelNames.LOCAL_B to Pose.LANDSCAPE_LEFT)
        assertEquals(1, log.add(answers, still = true).number)
        assertEquals(2, log.add(answers, still = false).number)
        assertEquals(2, log.undoLast()!!.number)
        assertEquals(2, log.nextNumber)
        assertTrue(log.toCsv().lines()[1].endsWith("landscape_left,portrait_up,landscape_left,true"))

        // the captured answer comes from the newest window alone, with no smoothing
        val sessions = Pose.entries.map { labeled(it, "train", "ann") }
        val data = LocalData.build(sessions, "all", "acc+grav")
        val m = SoftmaxModel(12).also { it.train(data.train.X, data.train.y, epochs = 30, rng = java.util.Random(1)) }
        val models = listOf(StoredModel(ModelNames.GLOBAL, m, "acc+grav", "run", 0L))
        val predictor = LivePredictor()
        predictor.update(session(Pose.SCREEN_UP, seconds = 2), models)                 // older window: another pose
        val frame = predictor.update(session(Pose.PORTRAIT_DOWN, seconds = 2), models)
        assertEquals(Pose.PORTRAIT_DOWN, frame.windowAnswers()[ModelNames.GLOBAL])
    }

    // ---------------- extra credit: secure aggregation (core/SecureAggregation.kt) ----------------

    @Test
    fun secureAggregationMasksCancelAndOneUploadAloneIsNoise() {
        // 1) both phones compute the same shared key from each other's public key
        val a = SecureSession("A", "B", "run1")
        val b = SecureSession("B", "A", "run1")
        a.agree(b.publicKey, 2)
        b.agree(a.publicKey, 2)

        // 2) each phone masks its own model
        val rng = java.util.Random(4)
        val modelA = SoftmaxModel(12).also { m -> for (i in m.W.indices) m.W[i] = rng.nextGaussian() * 5; for (i in m.b.indices) m.b[i] = rng.nextGaussian() }
        val modelB = SoftmaxModel(12).also { m -> for (i in m.W.indices) m.W[i] = rng.nextGaussian() * 5; for (i in m.b.indices) m.b[i] = rng.nextGaussian() }
        val maskedA = a.mask(modelA, 200, round = 3, attempt = 0)
        val maskedB = b.mask(modelB, 300, round = 3, attempt = 0)

        // 3) the server adds the two uploads (modulo 2^64): the masks cancel and FedAvg is left
        val expected = SoftmaxModel.average(listOf(modelA, modelB), listOf(200, 300))
        val flat = expected.W + expected.b
        for (i in flat.indices) {
            val sum = java.lang.Long.parseUnsignedLong(maskedA[i], 16) + java.lang.Long.parseUnsignedLong(maskedB[i], 16)
            assertEquals(flat[i], sum / (SecureAggregation.SCALE * 500), 1e-8)
        }

        // one upload alone looks nothing like the real weights
        val alone = SecureAggregation.decodeAlone(maskedA, 200)
        assertTrue(alone.take(20).all { abs(it) > 1e3 })
        // a retry of the same round uses a different mask
        assertTrue(maskedA != a.mask(modelA, 200, round = 3, attempt = 1))
    }

    @Test
    fun relayedLocalModelCanOnlyBeOpenedWithTheSharedKey() {
        val a = SecureSession("A", "B", "run1")
        val b = SecureSession("B", "A", "run1")
        val stranger = SecureSession("B", "A", "run1")   // same role, different private key (e.g. the server guessing)
        a.agree(b.publicKey, 2); b.agree(a.publicKey, 2); stranger.agree(a.publicKey, 2)
        val blob = a.encryptLocalModel("{\"W\":[1.5,-2.0]}")
        assertEquals("{\"W\":[1.5,-2.0]}", b.decryptOtherLocalModel(blob))
        assertEquals(null, stranger.decryptOtherLocalModel(blob))
        assertTrue(!blob["ciphertext"].toString().contains("1.5"))
    }

    // ---------------- recording screen: the label check ----------------

    @Test
    fun labelCheckCatchesAWrongLabel() {
        val rows = session(Pose.PORTRAIT_DOWN, seconds = 6)
        assertTrue(LabelCheck.matchFraction(rows, Pose.PORTRAIT_DOWN) > 0.95)
        assertTrue(LabelCheck.matchFraction(rows, Pose.PORTRAIT_UP) < 0.05)
    }

    @Test
    fun blindTestAnswerAveragesTheLastTwoWindows() {
        val sessions = Pose.entries.map { labeled(it, "train", "ann") }
        val data = LocalData.build(sessions, "all", "acc+grav")
        val m = SoftmaxModel(12).also { it.train(data.train.X, data.train.y, epochs = 30, rng = java.util.Random(1)) }
        val models = listOf(StoredModel(ModelNames.GLOBAL, m, "acc+grav", "run", 0L))
        val predictor = LivePredictor()

        // 3 s of Landscape Left: both windows agree
        assertEquals(Pose.LANDSCAPE_LEFT, predictor.captureAnswers(session(Pose.LANDSCAPE_LEFT, seconds = 3), models)[ModelNames.GLOBAL])

        // the newest 0.2 s is a bump that points the phone screen-up; averaged with the steady window
        // before it, the answer stays Landscape Left
        val rows = session(Pose.LANDSCAPE_LEFT, seconds = 3).map { r ->
            if (r[0] < 2800) r else r.copyOf().also { it[1] = 0.0; it[2] = 0.0; it[3] = 9.81; it[4] = 0.0; it[5] = 0.0; it[6] = 9.81 }
        }
        assertEquals(Pose.LANDSCAPE_LEFT, predictor.captureAnswers(rows, models)[ModelNames.GLOBAL])

        // not enough data: no answer (the screen then falls back to the newest window)
        assertTrue(predictor.captureAnswers(session(Pose.SCREEN_UP, seconds = 0), models).isEmpty())
    }

    // ---------------- extra credit: compressed secure aggregation (8-bit changes, 16-bit masks) ----------------

    @Test
    fun compressedMasksCancelWithinOneRoundingStep() {
        val a = SecureSession("A", "B", "run1")
        val b = SecureSession("B", "A", "run1")
        a.agree(b.publicKey, 2); b.agree(a.publicKey, 2)
        val rng = java.util.Random(7)
        val changeA = DoubleArray(78) { rng.nextDouble() - 0.5 }
        val changeB = DoubleArray(78) { rng.nextDouble() - 0.5 }
        val upA = a.maskCompressed(changeA, 0.6, 1.0, round = 4, attempt = 0)
        val upB = b.maskCompressed(changeB, 0.4, 1.0, round = 4, attempt = 0)
        assertEquals(208, upA.payload.length)                                 // 78 x 2 bytes, base64
        // the server adds the two uploads modulo 2^16: the masks cancel
        val da = java.util.Base64.getDecoder().decode(upA.payload)
        val db = java.util.Base64.getDecoder().decode(upB.payload)
        for (i in 0 until 78) {
            val va = (da[2 * i].toInt() and 0xff) or ((da[2 * i + 1].toInt() and 0xff) shl 8)
            val vb = (db[2 * i].toInt() and 0xff) or ((db[2 * i + 1].toInt() and 0xff) shl 8)
            val sum = ((va + vb) and 0xFFFF).toShort().toInt()
            val expected = 0.6 * changeA[i] + 0.4 * changeB[i]
            assertTrue(abs(sum / 127.0 - expected) <= 1.0 / 127 + 1e-12)     // at most one 8-bit step
        }
        // one upload alone is noise, and a change bigger than the scale is clipped (and counted)
        assertTrue(SecureAggregation.decodeCompressedAlone(upA.payload, 0.6, 1.0).any { abs(it) > 10 })
        assertEquals(1, a.maskCompressed(DoubleArray(78) { if (it == 0) 3.0 else 0.0 }, 0.5, 1.0, 4, 0).clipped)
    }

    // ---------------- pose actions (core/PoseActions.kt): 3 s to switch on, 1 s to switch off ----------------

    @Test
    fun poseActionSwitchesOnAfterThreeSecondsAndOffAfterOne() {
        val t = PoseActionTrigger()
        var now = 0L
        fun feed(pose: Pose, conf: Double, ms: Long): List<PoseActionTrigger.Change> {
            val out = ArrayList<PoseActionTrigger.Change>()
            val end = now + ms
            while (now < end) { t.update(pose, conf, now)?.let { out.add(it) }; now += 200 }
            return out
        }
        // 2.8 s upside down: not yet
        assertTrue(feed(Pose.PORTRAIT_DOWN, 0.95, 2800).isEmpty())
        assertTrue(t.holdProgress(now) > 0.8)
        // a moment later the flashlight switches on
        val on = feed(Pose.PORTRAIT_DOWN, 0.95, 600)
        assertEquals(PoseAction.FLASHLIGHT, (on.single() as PoseActionTrigger.Change.Started).action)
        // a short wobble (0.6 s) does not switch it off
        assertTrue(feed(Pose.PORTRAIT_UP, 0.95, 600).isEmpty())
        assertTrue(feed(Pose.PORTRAIT_DOWN, 0.95, 400).isEmpty())
        // 1 s in another pose switches it off
        val off = feed(Pose.PORTRAIT_UP, 0.95, 1200)
        assertEquals(PoseAction.FLASHLIGHT, (off.single() as PoseActionTrigger.Change.Stopped).action)
        assertEquals(null, t.active)
    }

    @Test
    fun poseActionNeedsAConfidentModelAndAPoseWithAnAction() {
        val t = PoseActionTrigger()
        for (i in 0..30) assertEquals(null, t.update(Pose.SCREEN_DOWN, 0.6, i * 200L))   // unsure: nothing
        for (i in 0..30) assertEquals(null, t.update(Pose.PORTRAIT_UP, 0.99, i * 200L))  // no action for Portrait Up
        assertEquals("A", PoseAction.DO_NOT_DISTURB.client)
        assertEquals("B", PoseAction.TIMER.client)
        assertEquals(2, PoseAction.entries.count { it.client == "A" })
    }

    // ---------------- on-device cost (core/DeviceCost.kt) ----------------

    @Test
    fun deviceCostCountsTimeBytesAndBattery() {
        val transport = object : Transport {
            override fun request(method: String, path: String, body: Map<String, Any?>?) = emptyMap<String, Any?>()
            override var bytesSent = 1000L
            override var bytesReceived = 5000L
        }
        var reading = BatteryReading(80, 3_000_000, charging = false)
        val tracker = CostTracker(transport) { reading }
        tracker.start()
        tracker.baselineTrained(40.0)
        tracker.addRound(10.0); tracker.addRound(20.0)
        transport.bytesSent += 3000; transport.bytesReceived += 7000
        reading = BatteryReading(79, 2_988_000, charging = false)
        val cost = tracker.finish(featureMsPerWindow = 0.05, predictMsPerWindow = 0.01)
        assertEquals(15.0, cost.trainMsPerRound, 1e-9)
        assertEquals(3000L, cost.bytesSent)
        assertEquals(7000L, cost.bytesReceived)
        assertEquals(12.0, cost.batteryUsedMah!!, 1e-9)
        assertEquals(2, cost.rounds)
        val upload = cost.toMap("A", "run1")
        assertEquals(12.0, upload["battery_used_mah"])
        assertEquals(false, upload["charging"])
        // timing three tiny models gives a small positive number
        val m = SoftmaxModel(12)
        assertTrue(CostTracker.timePredictions(listOf(m, m, m), listOf(DoubleArray(12))) > 0.0)
    }
}
