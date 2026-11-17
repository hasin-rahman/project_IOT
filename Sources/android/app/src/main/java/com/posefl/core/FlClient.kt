package com.posefl.core

import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URI

// =====================================================================================
// Transport: JSON over HTTP. Pure java.net, so it works on Android and in JVM tests.
// =====================================================================================

/** The server answered with an error status (e.g. 409 stale_version). */
class HttpError(val code: Int, val error: String, val body: Map<String, Any?>) : IOException("HTTP $code: $error")

interface Transport {
    /** Returns the parsed JSON object. Throws [HttpError] for 4xx/5xx and IOException if unreachable. */
    fun request(method: String, path: String, body: Map<String, Any?>? = null): Map<String, Any?>
}

class HttpTransport(baseUrl: String, private val connectTimeoutMs: Int = 4_000, private val readTimeoutMs: Int = 15_000) : Transport {
    val base = baseUrl.trim().trimEnd('/')

    init {
        require(base.startsWith("http://") || base.startsWith("https://")) {
            "Server URL must start with http://, e.g. http://192.168.1.10:8000"
        }
    }

    override fun request(method: String, path: String, body: Map<String, Any?>?): Map<String, Any?> {
        val conn = URI(base + path).toURL().openConnection() as HttpURLConnection
        try {
            conn.requestMethod = method
            conn.connectTimeout = connectTimeoutMs
            conn.readTimeout = readTimeoutMs
            conn.setRequestProperty("Content-Type", "application/json")
            if (body != null) {
                val bytes = Json.stringify(body).toByteArray(Charsets.UTF_8)
                conn.doOutput = true
                conn.setFixedLengthStreamingMode(bytes.size)
                conn.outputStream.use { it.write(bytes) }
            }
            val code = conn.responseCode
            val stream = if (code in 200..299) conn.inputStream else conn.errorStream
            val text = stream?.bufferedReader()?.use { it.readText() } ?: ""
            val json = if (text.isNotBlank()) Json.parseObject(text) else emptyMap()
            if (code !in 200..299) throw HttpError(code, json.str("error", "HTTP $code"), json)
            return json
        } finally {
            conn.disconnect()
        }
    }
}

// =====================================================================================
// Models on the wire and on disk
// =====================================================================================

object ModelIo {
    fun toMap(m: SoftmaxModel, featureSet: String): Map<String, Any?> = linkedMapOf(
        "n_features" to m.nFeatures, "n_classes" to m.nClasses, "feature_set" to featureSet,
        "W" to m.W, "b" to m.b,
    )

    fun fromMap(d: Map<String, Any?>): SoftmaxModel {
        val m = SoftmaxModel(d.int("n_features"), d.int("n_classes", Pose.COUNT))
        val W = d.doubles("W")
        val b = d.doubles("b")
        require(W.size == m.W.size && b.size == m.b.size) { "model has the wrong number of parameters" }
        W.copyInto(m.W); b.copyInto(m.b)
        return m
    }
}

/** A model plus the feature set it expects and where it came from. */
class StoredModel(val name: String, val model: SoftmaxModel, val featureSet: String, val runId: String, val savedAt: Long)

/** Keeps the latest local-A, local-B and global models on the phone for the live demo screen. */
class ModelStore(private val dir: File) {
    init { dir.mkdirs() }

    fun save(name: String, model: SoftmaxModel, featureSet: String, runId: String) {
        val d = LinkedHashMap(ModelIo.toMap(model, featureSet))
        d["run_id"] = runId
        d["saved_at"] = System.currentTimeMillis()
        val tmp = File(dir, "$name.json.tmp")
        tmp.writeText(Json.stringify(d))
        tmp.renameTo(File(dir, "$name.json"))
    }

    fun load(name: String): StoredModel? = try {
        val d = Json.parseObject(File(dir, "$name.json").readText())
        StoredModel(name, ModelIo.fromMap(d), d.str("feature_set", FeatureSets.BASELINE), d.str("run_id"), d.dbl("saved_at").toLong())
    } catch (e: Exception) {
        null
    }

    /** local_A, local_B and global (whichever exist), in that order. */
    fun loadAll(): List<StoredModel> = listOf("local_A", "local_B", "global").mapNotNull { load(it) }
}

// =====================================================================================
// Local data: raw sessions on the phone -> feature datasets for one feature set
// =====================================================================================

class LabeledSession(val id: String, val pose: Pose, val split: String, val participant: String, val rows: List<DoubleArray>)

/** Feature vectors [X] with their pose labels [y] (pose index 0..5). Lives only on the phone. */
class Dataset(val X: List<DoubleArray>, val y: IntArray) {
    val size: Int get() = y.size
    fun isEmpty() = y.isEmpty()

    /** How many examples there are of each pose, e.g. [210, 205, 215, 0, 0, 0] for client A. */
    fun classCounts(): IntArray {
        val counts = IntArray(Pose.COUNT)
        for (label in y) counts[label]++
        return counts
    }
}

class LocalData(
    val featureSet: String,
    val train: Dataset,
    val validation: Dataset,
    val test: Dataset,
    /** Test sessions from people who recorded no training data on this phone (unseen users). */
    val testUnseen: Dataset,
    val trainParticipants: Set<String>,
) {
    /** Named test sets that have data, in report order. */
    fun testSets(): List<Pair<String, Dataset>> =
        listOf("val" to validation, "test" to test, "test_unseen" to testUnseen).filter { !it.second.isEmpty() }

    companion object {
        /**
         * Training data is ONLY this client's train sessions of its own three poses (the non-IID split).
         * Validation and test sessions may contain all six poses and are never used for training.
         */
        fun build(sessions: List<LabeledSession>, clientId: String, featureSet: String, cfg: WindowConfig = WindowConfig()): LocalData {
            val trainPoses = Pose.trainPosesFor(clientId).toSet()
            val trainSessions = sessions.filter { it.split == "train" && it.pose in trainPoses }
            val trainPeople = trainSessions.map { it.participant }.toSet()

            fun ds(list: List<LabeledSession>): Dataset {
                val X = ArrayList<DoubleArray>()
                val y = ArrayList<Int>()
                for (s in list) {
                    val f = FeatureExtractor.extract(s.rows, featureSet, cfg)
                    X.addAll(f)
                    repeat(f.size) { y.add(s.pose.index) }
                }
                return Dataset(X, y.toIntArray())
            }

            val tests = sessions.filter { it.split == "test" }
            return LocalData(
                featureSet,
                ds(trainSessions),
                ds(sessions.filter { it.split == "val" }),
                ds(tests),
                ds(tests.filter { it.participant !in trainPeople }),
                trainPeople,
            )
        }
    }
}

fun Metrics.toMap(): Map<String, Any?> = linkedMapOf(
    "accuracy" to accuracy, "macro_f1" to macroF1, "n" to total,
    "per_class_recall" to perClassRecall, "per_class_f1" to perClassF1,
    "confusion" to confusion.map { it.toList() },
)

fun SoftmaxModel.metricsOn(d: Dataset): Metrics = Metrics(d.y, IntArray(d.size) { predict(d.X[it]) })

// =====================================================================================
// The federated-learning client (same protocol as server/sim_clients.py)
// =====================================================================================

/** One evaluation result shown on the phone after a run. */
class EvalRow(val modelName: String, val testSet: String, val metrics: Metrics)

interface FlListener {
    fun log(message: String)
    /** phase: "connecting", "baseline", "training", "waiting", "evaluating", "done", "idle", "error" */
    fun progress(runId: String, version: Int, rounds: Int, phase: String, lastValAcc: Double?)
    fun results(runId: String, featureSet: String, rows: List<EvalRow>)
    /** Called with the exact JSON text of every upload, so the app can show what leaves the phone. */
    fun uploaded(endpoint: String, json: String) {}
}

/** The training settings the server hands out, so both phones train the same way. */
class TrainConfig(json: Map<String, Any?>) {
    val rounds = json.int("rounds")
    val featureSet = json.str("feature_set", FeatureSets.BASELINE)
    val localEpochs = json.int("local_epochs", 1)
    val baselineEpochs = json.int("baseline_epochs", 20)
    val lr = json.dbl("lr", 0.1)
    val batchSize = json.int("batch_size", 16)
    val l2 = json.dbl("l2", 1e-4)
    val seed = json.int("seed", 0)
    /** Extra credit: upload masked weights that only make sense once the server adds both phones' uploads. */
    val secure = json.bool("secure_aggregation")
}

/** The server's answer to GET /global: where the run is, the current global model and the settings. */
class ServerState(json: Map<String, Any?>) {
    val runId = json.str("run_id")
    /** How many times the server has averaged so far = the round number we are in. */
    val version = json.int("version")
    /** True when all rounds are finished. */
    val done = json.bool("done")
    /** Clients whose update for this round has already arrived. */
    val submitted = json.list("submitted")
    /** Secure aggregation: how often this round was restarted (a new mask is used every time). */
    val attempt = json.int("attempt")
    /** Secure aggregation: changes whenever a phone publishes a new public key. */
    val keyEpoch = json.int("key_epoch")
    val globalModel: SoftmaxModel by lazy { ModelIo.fromMap(json.obj("model")) }
    val config = TrainConfig(json.obj("train_config"))
}

/**
 * The client side of federated learning. [run] repeats these steps:
 *
 *  1. [register] with the server, then ask it for the current state (GET /global).
 *  2. New run? [startNewRun]: build features from this phone's recordings and
 *     [trainLocalOnlyModel] FROM SCRATCH (the independent baseline), upload it.
 *  3. [federatedRound]: take the global model -> train a copy on local data -> upload the weights.
 *     The server averages when both phones have sent theirs (FedAvg), and the loop repeats.
 *  4. After the last round, [evaluateAllModels]: score local A, local B and the global model
 *     on this phone's validation / test recordings, and save the three models for the live screen.
 *
 * Network failures never stop the loop: it waits and retries (dropout recovery).
 * PRIVACY: only model weights, sample counts and scores are sent. Raw data and features stay here.
 * With secure aggregation switched on for a run, even the weights are masked ([agreeOnKeys],
 * core/SecureAggregation.kt), so the server never sees a single phone's model.
 */
class FlRunner(
    private val transport: Transport,
    private val clientId: String,
    private val device: String,
    private val participant: String,
    private val dataFor: (String) -> LocalData,
    private val store: ModelStore?,
    private val listener: FlListener,
    private val sleeper: (Long) -> Unit = { Thread.sleep(it) },
) {
    @Volatile var stopRequested = false
    private val otherId = if (clientId == "A") "B" else "A"

    // ---- what this phone remembers about the run it is taking part in ----
    private var currentRun: String? = null
    private var data: LocalData? = null
    /** This phone's local-only baseline. Never mixed with the global model. */
    private var localModel: SoftmaxModel? = null
    private var evaluated = false
    private var rng = java.util.Random(0)
    private var lastValAcc: Double? = null
    private var registered = false
    private var failures = 0
    private var waitingLogged = false
    /** Secure aggregation: this phone's keys for the current run (null when the run is not secure). */
    private var secure: SecureSession? = null
    private var keyWaitLogged = false
    private var lastAttempt = 0
    /** The last update of this phone the way the server received it (a real model, or noise when masked). */
    private var lastUploadAsSeenByServer: SoftmaxModel? = null

    /** Blocks until stopped (or, if [followNewRuns] is false, until one run has been evaluated). */
    fun run(followNewRuns: Boolean) {
        while (!stopRequested) {
            try {
                if (!registered) register()
                val server = ServerState(transport.request("GET", "/global?client_id=$clientId"))
                noteReconnected()

                if (server.runId != currentRun && !startNewRun(server)) return

                // secure aggregation: nothing can be masked until both phones have exchanged public keys
                val session = secure
                if (session != null && !agreeOnKeys(session, server) && !server.done) {
                    waitForOtherKey(server)
                    continue
                }

                when {
                    server.done -> {
                        if (!evaluated) {
                            evaluateAllModels(server)
                            evaluated = true
                            if (!followNewRuns) return
                            listener.log("Waiting for the server to start a new run (Stop to finish).")
                        }
                        listener.progress(server.runId, server.version, server.config.rounds, "idle", lastValAcc)
                        sleeper(1000)
                    }
                    server.submitted.contains(clientId) -> waitForOtherPhone(server)
                    else -> federatedRound(server)
                }
            } catch (e: HttpError) {
                listener.log("Server error ${e.code}: ${e.error}")
                listener.progress(currentRun ?: "-", 0, 0, "error", null)
                if (e.code in 400..499 && e.code != 409 && e.code != 404) return
                sleeper(2000)
            } catch (e: IOException) {
                // Wi-Fi lost or server not reachable: keep trying (dropout recovery)
                failures++
                if (failures == 1 || failures % 10 == 0) listener.log("Server unreachable (${e.message ?: e.javaClass.simpleName}); retrying every 2 s...")
                listener.progress(currentRun ?: "-", 0, 0, "connecting", lastValAcc)
                sleeper(2000)
            }
        }
        listener.log("Stopped.")
    }

    // ------------------------------------------------------------------ step 1: connect

    private fun register() {
        listener.progress(currentRun ?: "-", 0, 0, "connecting", null)
        upload("/register", mapOf("client_id" to clientId, "device" to device, "participant" to participant))
        registered = true
    }

    private fun noteReconnected() {
        if (failures > 0) listener.log("Reconnected to the server.")
        failures = 0
    }

    // ------------------------------------------------------------------ step 2: new run + local baseline

    /** Prepares the data for a new run and trains the local-only model. Returns false if there is no data. */
    private fun startNewRun(server: ServerState): Boolean {
        val cfg = server.config
        listener.log("Run ${server.runId}: feature set ${cfg.featureSet}, ${cfg.rounds} rounds, ${cfg.localEpochs} local epoch(s), seed ${cfg.seed}")
        listener.progress(server.runId, server.version, cfg.rounds, "baseline", null)

        val d = dataFor(cfg.featureSet)
        if (d.train.isEmpty()) {
            listener.log("No training windows for client $clientId. Record train sessions of ${Pose.trainPosesFor(clientId).joinToString { it.label }} first.")
            listener.progress(server.runId, server.version, cfg.rounds, "error", null)
            return false
        }
        listener.log("Local data: ${d.train.size} train, ${d.validation.size} validation, ${d.test.size} test windows" +
            (if (d.testUnseen.size > 0) " (${d.testUnseen.size} from unseen users)" else ""))
        listener.log("Train windows per pose: " + describeCounts(d.train.classCounts()))

        rng = java.util.Random(cfg.seed * 7919L + clientId.hashCode())
        val local = trainLocalOnlyModel(d, cfg)
        if (cfg.secure) {
            // publish a fresh public key; the local model is uploaded later, encrypted (see agreeOnKeys)
            val session = SecureSession(clientId, otherId, server.runId)
            upload("/pubkey", linkedMapOf("client_id" to clientId, "run_id" to server.runId, "public_key" to session.publicKey))
            secure = session
            listener.log("Secure aggregation is ON for this run: weights are masked before upload.")
        } else {
            secure = null
            upload("/local_model", linkedMapOf(
                "client_id" to clientId, "model" to ModelIo.toMap(local, cfg.featureSet),
                "num_samples" to d.train.size, "class_counts" to d.train.classCounts()))
        }
        store?.save("local_$clientId", local, cfg.featureSet, server.runId)

        data = d; localModel = local; currentRun = server.runId
        evaluated = false; lastValAcc = null; waitingLogged = false
        keyWaitLogged = false; lastAttempt = 0; lastUploadAsSeenByServer = null
        return true
    }

    // ------------------------------------------------------------------ secure aggregation: shared key

    /**
     * Makes sure this phone has the shared key (Diffie-Hellman): it downloads the other phone's
     * public key from the server and combines it with its own private key. Returns false while the
     * other phone has not published its key yet. Afterwards the local-only model is uploaded
     * encrypted, so the server can relay it but cannot read it.
     */
    private fun agreeOnKeys(session: SecureSession, server: ServerState): Boolean {
        if (session.ready && session.keyEpoch == server.keyEpoch) return true
        val reply = try {
            transport.request("GET", "/pubkey?client_id=$otherId")
        } catch (e: HttpError) {
            if (e.code == 404) return false else throw e
        }
        session.agree(reply.str("public_key"), reply.int("key_epoch"))
        listener.log("Agreed on a shared key with client $otherId. The server only relayed public keys and cannot compute it.")
        val d = data!!
        val localJson = Json.stringify(ModelIo.toMap(localModel!!, server.config.featureSet))
        upload("/local_model", linkedMapOf(
            "client_id" to clientId, "encrypted_model" to session.encryptLocalModel(localJson),
            "num_samples" to d.train.size, "class_counts" to d.train.classCounts()))
        return true
    }

    private fun waitForOtherKey(server: ServerState) {
        listener.progress(server.runId, server.version, server.config.rounds, "waiting", lastValAcc)
        if (!keyWaitLogged) { listener.log("Waiting for client $otherId to publish its public key."); keyWaitLogged = true }
        sleeper(500)
    }

    /**
     * The independent baseline: a brand-new model (all weights 0) trained ONLY on this phone's own
     * training data. It never starts from, and is never mixed with, the global model. Because it
     * only ever sees this client's three poses, it cannot recognise the other client's three.
     */
    private fun trainLocalOnlyModel(d: LocalData, cfg: TrainConfig): SoftmaxModel {
        val model = SoftmaxModel(FeatureSets.size(cfg.featureSet))
        val t0 = System.currentTimeMillis()
        model.train(d.train.X, d.train.y, epochs = cfg.baselineEpochs, lr = cfg.lr, batchSize = cfg.batchSize, l2 = cfg.l2, rng = rng)
        listener.log("Trained local-only model ($clientId) from scratch in ${System.currentTimeMillis() - t0} ms; uploading it so the other phone can compare.")
        return model
    }

    // ------------------------------------------------------------------ step 3: one federated round

    /** Download the global model, train a copy on this phone's data, upload the new weights. */
    private fun federatedRound(server: ServerState) {
        val cfg = server.config
        val d = data!!
        val round = server.version
        waitingLogged = false
        listener.progress(server.runId, round, cfg.rounds, "training", lastValAcc)

        // a) how good is the current global model on this phone's validation recordings?
        val global = server.globalModel
        val valAcc = if (d.validation.isEmpty()) null else global.evaluate(d.validation.X, d.validation.y).accuracy

        // b) local training: start from the global model, train on this phone's data only
        val model = global.copy()
        val t0 = System.currentTimeMillis()
        model.train(d.train.X, d.train.y, epochs = cfg.localEpochs, lr = cfg.lr, batchSize = cfg.batchSize, l2 = cfg.l2, rng = rng)
        val trainMs = System.currentTimeMillis() - t0
        val fit = model.evaluate(d.train.X, d.train.y)

        // c) upload: the weights plus metadata. No sensor readings and no feature vectors.
        val update = linkedMapOf<String, Any?>(
            "client_id" to clientId, "run_id" to server.runId, "base_version" to round,
            "num_samples" to d.train.size, "class_counts" to d.train.classCounts(),
            "train_loss" to round5(fit.loss), "train_acc" to round5(fit.accuracy),
            "train_ms" to trainMs, "val_acc" to valAcc?.let { round5(it) },
        )
        val session = secure
        val seenByServer: SoftmaxModel
        if (session == null) {
            update["model"] = ModelIo.toMap(model, cfg.featureSet)
            seenByServer = model
        } else {
            // secure aggregation: upload the masked numbers instead of the real weights
            if (server.attempt != lastAttempt && server.attempt > 0) {
                listener.log("Round $round: the server restarted this round (a masked update cannot be used on its own). Sending again with a new mask.")
            }
            lastAttempt = server.attempt
            val masked = session.mask(model, d.train.size, round, server.attempt)
            update["masked_model"] = linkedMapOf(
                "n_features" to model.nFeatures, "n_classes" to model.nClasses,
                "feature_set" to cfg.featureSet, "values" to masked)
            update["attempt"] = server.attempt
            update["key_epoch"] = session.keyEpoch
            seenByServer = modelFromNumbers(SecureAggregation.decodeAlone(masked, d.train.size), model)
        }
        try {
            val reply = upload("/update", update)
            lastValAcc = valAcc
            lastUploadAsSeenByServer = seenByServer
            listener.log("Round $round: sent ${model.numParams} " + (if (session == null) "weights" else "MASKED weights") +
                (valAcc?.let { String.format(java.util.Locale.US, " (global v$round val acc %.1f%%)", it * 100) } ?: "") +
                if (reply.bool("aggregated")) " -> server averaged v${reply.int("version")}" else "")
        } catch (e: HttpError) {
            // 409 = the server closed this round without us (we were offline). Just fetch the newest model.
            if (e.code == 409) listener.log("Round $round: server already moved on (${e.error}); downloading the newest model.")
            else throw e
        }
    }

    private fun waitForOtherPhone(server: ServerState) {
        listener.progress(server.runId, server.version, server.config.rounds, "waiting", lastValAcc)
        if (!waitingLogged) { listener.log("Round ${server.version}: waiting for client $otherId"); waitingLogged = true }
        sleeper(250)
    }

    // ------------------------------------------------------------------ step 4: compare the three models

    private fun evaluateAllModels(server: ServerState) {
        val runId = server.runId
        val spec = server.config.featureSet
        val d = data!!
        listener.progress(runId, server.version, server.config.rounds, "evaluating", null)

        val global = server.globalModel
        store?.save("global", global, spec, runId)
        val other = downloadOtherLocalModel()
        if (other != null) store?.save("local_$otherId", other, spec, runId)
        else listener.log("Client $otherId has not uploaded a local model; comparing two models only.")

        val models = linkedMapOf<String, SoftmaxModel?>("local_A" to null, "local_B" to null, "global" to global)
        models["local_$clientId"] = localModel
        models["local_$otherId"] = other
        // for the privacy comparison: this phone's last upload, read the way the server received it
        models[SERVER_SEES] = lastUploadAsSeenByServer

        // every model is scored on the SAME recordings; only the scores are uploaded
        val rows = ArrayList<EvalRow>()
        for ((setName, set) in d.testSets()) {
            for ((modelName, model) in models) {
                if (model == null) continue
                val metrics = model.metricsOn(set)
                rows.add(EvalRow(modelName, setName, metrics))
                upload("/metrics", mapOf(
                    "client_id" to clientId, "model_name" to modelName, "test_set" to setName, "feature_set" to spec,
                    "participant" to participant, "metrics" to metrics.toMap()))
            }
            listener.log("Results on $setName: " + rows.filter { it.testSet == setName }.joinToString(" | ") {
                String.format(java.util.Locale.US, "%s %.1f%%", it.modelName, it.metrics.accuracy * 100)
            })
        }
        if (rows.isEmpty()) listener.log("No validation or test sessions recorded on this phone, so nothing to evaluate.")

        upload("/done", mapOf("client_id" to clientId, "run_id" to runId))
        listener.results(runId, spec, rows)
        listener.progress(runId, server.version, server.config.rounds, "done", null)
        listener.log("Run $runId finished. Models saved on the phone for the live screen.")
    }

    /** The other phone uploads its local-only model at the start of the run; the server relays it. */
    private fun downloadOtherLocalModel(): SoftmaxModel? {
        for (attempt in 0 until 20) {
            try {
                val reply = transport.request("GET", "/local_model?client_id=$otherId")
                if (reply.containsKey("model")) return ModelIo.fromMap(reply.obj("model"))
                // secure aggregation: the server relayed an encrypted model that only this phone can open
                val json = secure?.decryptOtherLocalModel(reply.obj("encrypted_model"))
                if (json != null) return ModelIo.fromMap(Json.parseObject(json))
            } catch (e: HttpError) {
                if (e.code != 404) throw e
            }
            if (stopRequested) return null
            sleeper(500)
        }
        return null
    }

    /** Builds a model with the same shape as [like] from one flat array of numbers (W then b). */
    private fun modelFromNumbers(numbers: DoubleArray, like: SoftmaxModel): SoftmaxModel {
        val m = SoftmaxModel(like.nFeatures, like.nClasses)
        numbers.copyInto(m.W, 0, 0, m.W.size)
        numbers.copyInto(m.b, 0, m.W.size, numbers.size)
        return m
    }

    companion object {
        /** Result name for "one upload of this phone, as the server sees it". */
        const val SERVER_SEES = "server_sees"
    }

    // ------------------------------------------------------------------ helpers

    /** Every upload goes through here, so the app can display exactly what left the phone. */
    private fun upload(endpoint: String, body: Map<String, Any?>): Map<String, Any?> {
        val reply = transport.request("POST", endpoint, body)
        listener.uploaded(endpoint, Json.stringify(body))
        return reply
    }

    private fun describeCounts(counts: IntArray): String =
        Pose.entries.joinToString(", ") { "${it.label} ${counts[it.index]}" }

    private fun round5(v: Double) = Math.round(v * 1e5) / 1e5
}
