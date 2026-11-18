# Demo-day guide: Federated Phone-Pose Recognition

CSC 8223 · Hasin Rahman and Shiv Brahmbhatt · written Nov 13, 2026 from the project handout, our proposal and the
demo-day slides in `Presentation guidlines/`; updated Nov 17 for the combined app (version 0.6), with Shiv's Q&A
answers from his FedPose version (section 7); updated Nov 18 for version 0.7 (curious-server attack, cost on the
phones, Pose actions, Compare runs; FedPose import removed).

## 1. What demo day looks like (from the slides)

Before presenting: server and both phones running, all three models loaded.

| Part | Time | What happens |
|---|---|---|
| Presentation with live demo | 8 min | While the system runs: both phones collecting and training, what gets uploaded, independent local models, our contribution |
| Q&A | 5 min | Questions on the core findings. Every member answers on their own part |
| Blind test (4–5 teams only) | 7 min | Judges place the phone 36 times (each pose 6 times) following a pose card. The app records one window and shows all three predictions. We may not touch the phones |

The demo is 80% of the project grade and is scored out of 100:

| Part | Points | How it is scored | Where our system covers it |
|---|---|---|---|
| A1 | 30 | 30 × global model accuracy over all 36 placements | Live screen → Blind test |
| A2 | 5 | 5 × local model A accuracy on A's 3 poses (18 placements) | same screen, "Local A" line |
| A3 | 5 | 5 × local model B accuracy on B's 3 poses (18 placements) | same screen, "Local B" line |
| B · Live on both phones | 11 | Both phones collect data and train locally, on site (one phone only: 6; pre-recorded: 3) | Record data + Federated training on both phones |
| B · Privacy | 13 | We show the real upload or server log (explained without evidence: 7) | Dashboard "What the phones upload", phone "Show what this phone uploaded", server console |
| B · Independent baselines | 11 | Local models near 0 on unseen poses, and we can explain why (cannot explain: 7) | Dashboard "Recall per pose", phone results "A poses / B poses" |
| C · Contribution | 5 + 10 | What we built and why; integrated and shown live | Sensor study, started live from the dashboard |
| C · Findings and Q&A | 5 + 5 | Explain the core findings; every member answers on their part | Section 4 and 5 below |

Ranking: teams in the blind test come first by total score, the rest follow by B + C. The slides also say each local
model **should get the other client's poses wrong**, and the judges check this.

## 2. Eight-minute plan

| Time | Who | Do | Say |
|---|---|---|---|
| 0:00 | Hasin | Dashboard on the projector | Two phones, six poses, A trains on three and B on the other three. Only weights go to the server. |
| 0:40 | both | Each phone: Record data → Train → one 10 s session (A: Screen Up, B: Landscape Left) → Keep | Each phone records its own data; the label check compares gravity with the chosen pose. |
| 1:50 | Hasin | Dashboard: **Start a new run**. Both phones: Federated training → **Start** | Each phone first trains its own model from zero, then joins 20 FedAvg rounds. Watch the round counter and the accuracy curve. |
| 3:00 | Hasin | Dashboard: **What the phones upload** → show the raw JSON. Phone: **Show what this phone uploaded** | This is the whole upload: 78 weights, the sample count, counts per pose, the loss. The server refuses any other field. |
| 4:00 | Shiv | Dashboard: **Recall per pose** table | Local A is about 0% on B's poses: it started from zero and only ever saw A's three poses. The global model gets all six. |
| 5:00 | Shiv | Dashboard form: start a run with another feature set (for example `acc`), both phones follow | Our contribution: the sensor study. Same pipeline, four sensor combinations. Show the result figure. |
| 6:40 | Hasin | Start the final run with the best feature set, then open the live screen | Findings (section 4). |

Finish on the feature set we want for the blind test: the live screen uses the models of the last finished run.

## 3. How the code works, in the order the data flows

Each row is one thing to be able to explain. File paths are under `Sources/android/app/src/main/java/com/posefl/`.

| Step | Where | In plain words |
|---|---|---|
| Read sensors | `app/SensorRecorder.kt` → `onSensorChanged` | Four sensors fire independently at about 50 Hz. We keep the latest value of each and write one row per accelerometer event: time, accelerometer, gravity, gyroscope, rotation. |
| Record a labelled session | `app/RecordActivity.kt` | Choose split and pose, 3 beeps, record N seconds, then a label check (`Pose.guessFromGravity`) warns if the phone was not in the chosen pose. Train sessions are limited to the client's three poses. |
| Cut into windows | `core/Features.kt` → `windowRanges` | 1-second windows, a new one every 0.5 s. First and last second dropped (the phone is still moving). |
| Make features | `core/Features.kt` → `windowFeatures` | Per sensor block: mean of x, y, z (which way the phone points) and standard deviation of x, y, z (how much it shakes). Baseline `acc+grav` = 12 numbers. |
| The model | `core/SoftmaxModel.kt` → `predictProba`, `train`, `sgdStep` | One linear layer + softmax: 6 scores from 12 features, 78 parameters. Trained with mini-batch SGD: error = predicted probability − correct answer; move each weight a small step against the average error × feature. |
| Non-IID split | `core/FlClient.kt` → `LocalData.build` | Training data = this client's train sessions of its own three poses only. Validation and test sessions have all six and are never trained on. |
| Local-only baseline | `core/FlClient.kt` → `trainLocalOnlyModel` | A new model with all weights 0, trained only on this phone's data. Never touches the global model. |
| One federated round | `core/FlClient.kt` → `federatedRound` | Download the global model → train a copy on local data → upload the weights and metadata. |
| Averaging | `Sources/server/fedavg.py` → `fedavg` | New global weight = sum of (client weight × client's share of the training windows). Runs when both updates are in (`FLState.update` → `aggregate` in `fl_server.py`). |
| Privacy rule | `Sources/server/fl_server.py` → `ALLOWED_UPLOAD_FIELDS`, `check_upload_fields`, `record_upload` | The list of fields a phone may send. Anything else is refused. Every upload is printed, saved and shown on the dashboard. |
| Compare three models | `core/FlClient.kt` → `evaluateAllModels`, `core/Metrics.kt` | The server relays the other phone's local model. Each phone scores all three models on the same recordings and uploads only the scores. |
| Live prediction | `core/Live.kt` → `LivePredictor.update` | Newest 1-second window → same features → all three models. The live display averages the last 4 windows. |
| Blind test | `app/PredictActivity.kt` → `startCapture`, `captureWhenStill`, `capture`; `core/Live.kt` → `captureAnswers` | Tap → 3 beeps → wait until still for 1 s → each model's probabilities averaged over the last two 1-second windows → show and log the three answers. (Waiting for stillness is from Hasin's version, averaging windows from Shiv's.) |
| Practice score | `core/Live.kt` → `LiveTest.rubricPoints` | 30 × global accuracy + 5 × local A on A's poses + 5 × local B on B's poses. |
| Dropout recovery | `fl_server.py` → `tick`; `FlClient.kt` → `run` (the `IOException` branch and the 409 case) | The server closes a round after 30 s with whatever arrived. A phone that was offline gets "stale", downloads the newest model and continues. |
| Cost on the phone | `core/DeviceCost.kt` → `CostTracker` (`start`, `addRound`, `finish`, `timePredictions`); called from `FlClient.kt` | A stopwatch for one run: training time per round, feature and prediction time per window, bytes sent and received, battery from the charge counter. Sent as `POST /device_cost`. |
| Curious-server attack | `server/curious_server.py` → `bias_change_seen_by_server`, `guess_poses`, `score_guess`, `attack_run` | For each upload: bias in the upload − bias of the model the server sent; "went up" = the phone has that pose. Scored out of 6 against the handout's split. |
| Compare runs | `server/run_summary.py` → `summarize_run`; dashboard section 5 | Reads each run folder and makes one row: mode, accuracies, curious-server score, upload size, phone cost. |
| Pose actions | `core/PoseActions.kt` → `PoseActionTrigger.update`; `app/PoseActionsActivity.kt` → `decide`, `switchOn`, `switchOff` | Sense → decide → act. On after the same pose for 3 s at ≥75% confidence, off after 1 s out of it. Global / Local A / Local B switch. |

## 3b. Pose actions moment (30 seconds, after the live screen)

1. Phone A → **Pose actions** → select **Local A**, turn the phone upside down and hold it: nothing switches on
   (Local A never saw Portrait Down; the screen shows what it thinks instead).
2. Select **Global**, same pose: after 3 s the **flashlight** turns on. "Phone B taught this pose, and its data never
   left phone B."
3. Optional: **Local B** + screen down → no Do Not Disturb; **Global** → Do Not Disturb on.

Before the demo: grant *Do Not Disturb access* once (button on the screen), and make sure the last finished run's
models are on the phone.

## 4. Core findings to state

First real run (Nov 17, run_20261117_171543, both S24s, plain FedAvg, acc+grav, 20 rounds): global model 100% on
both phones' test and validation recordings; local A 49.9–50.0% and local B 50.0–51.2% (100% on their own poses, 0%
on the other phone's). Phone B's validation and test sets were small (about 85 windows); record more, and add an
unseen-user test, before quoting final numbers.

1. **Federation gives what local training cannot.** Each local model only knows three poses, so on a six-pose test it
   is right about half the time (about 100% on its own poses, about 0% on the others). The global model has seen
   neither phone's data, yet recognises all six. First real run: local about 50%, global 100%.
2. **Why a local model is near 0 on unseen poses.** It starts at zero and every training example says "this is one
   of my three poses", so training only ever pushes the scores of the other three poses down. Nothing ever pushes
   them up, so the model never answers them.
3. **Why FedAvg still works with completely different data.** Each phone pulls the shared weights toward its own
   three poses; averaging by sample count keeps both halves. The poses differ mainly in which axis gravity points
   along, which a linear model separates easily, so the two halves do not fight.
4. **Privacy.** A 20-second recording is about 1,000 sensor rows; an upload is 78 numbers plus counts (about 2 kB),
   whatever the amount of data.
5. **Sensor study (contribution).** State which of `acc`, `acc+grav`, `acc+grav+gyro`, `acc+grav+gyro+rot` generalised
   best to the unseen user, and what adding the gyroscope and rotation vector changed.

## 5. Who answers what (from the task division in our proposal)

| Topic | Answers |
|---|---|
| Sensors, sampling, windows, features, recording screen and label check | Hasin |
| Server: rounds, FedAvg, storing and relaying models, privacy allow-list and upload log, cloud with HTTPS | Hasin |
| Sensor/feature variations (how the feature sets are built) | Hasin |
| On-device model and training (softmax, SGD), weight upload and download | Shiv |
| Evaluation: metrics, three-model comparison, analysis script and plots | Shiv |
| Blind-test answer averaged over two windows; compression | Shiv (his ideas, combined into the app) |
| Curious-server attack and the `class_counts` leak we found | Hasin (with secure aggregation) |
| Cost on the phones (training, prediction, data, battery) | Shiv (with the on-device model) |
| Pose actions (the IoT loop) | both |
| Sensor study results and what they mean | both |
| Two versions built in parallel and combined (learning outcomes) | both |

(This follows the task division in our proposal; change it if you split it differently.)

Likely questions and short answers:

* *Is the local model really independent?* Yes: `trainLocalOnlyModel` creates a new all-zero model and trains it only
  on `d.train`, which `LocalData.build` limits to this client's poses. The global model is trained on a separate copy.
* *What exactly is uploaded?* Open the raw JSON on the dashboard: `W` (72 numbers), `b` (6), `num_samples`,
  `class_counts`, `train_loss`, `train_acc`, `train_ms`, `val_acc`, plus ids. Scores at the end. Nothing else is accepted.
* *Why softmax regression and not a neural network?* The poses are separated by the direction of gravity, which is
  linear in our features; 78 parameters train in milliseconds on the phone and keep uploads tiny.
* *Why 1-second windows with 50% overlap?* Long enough to average out hand tremor, short enough for a fast answer;
  overlap gives more training examples. Splits are made by session before windowing, so overlapping windows never
  leak between train and test.
* *Why weight by sample count?* A phone with more data should count for more; with equal data it is a plain average.
* *What if a phone drops out?* The round closes after the timeout with one update; the late phone is told its update
  is stale and resynchronises.

## 6. Extra credit: private, efficient federated learning on IoT devices (+3 to 5 minutes)

**One extension, evaluated three ways** (the handout scores one substantial, well-evaluated extension above several
small add-ons, out of 40: ambition 10, integration 10, experiments 10, analysis/demo 10):

| Question | How we answer it | Where |
|---|---|---|
| Does the server learn less? | **Curious-server attack**: from one plain upload the server names each phone's 3 poses (6/6 every round); with secure aggregation it is at chance (about 3/6). We also found and fixed our own leak: `class_counts` revealed the poses even in secure mode. | `curious_server.py`; dashboard card *Curious server*, section 3 |
| Is the model as good? | Global accuracy in plain, secure and compressed mode | dashboard section 5 *Compare runs*; `analyze_run.py` |
| Is it practical on a phone? | **Cost on the phones**: time per round, prediction time, bytes per round (compressed ≈ 4× smaller), battery | phone *Cost on this phone*; dashboard section 4 |
| Does it survive a phone dropping out? | Dropout: plain averages without it, secure discards and repeats the round | dashboard *Training* (discarded rounds) |

**Pose actions** are not claimed as extra credit on their own: they show what the models are for (an IoT device that
acts on what it senses) and make "the global model knows poses my phone never trained on" visible in the demo.

Our proposal listed three small extras: 8-bit update compression, adaptive participation and dropout recovery.
We built them in two versions (Shiv: compression, adaptive participation, dropout; Hasin: secure aggregation,
dropout) and combined them into one extension, because the handout says several small add-ons do not earn more than
one substantial, well-evaluated one. Adaptive participation was left out: with two phones, skipping one leaves a
single client, and the bytes saved are tiny. The report should state these changes.

### What it does, in plain words

Normally the server sees each phone's weights. With **secure aggregation** it sees only scrambled numbers and still
gets exactly the same average. **Compressed** secure aggregation does the same with uploads about a quarter of the size.

1. **Shared secret.** Each phone makes a private number and sends the matching public key through the server.
   Each phone combines its private number with the other's public key and both arrive at the same secret
   (Diffie-Hellman). The server only passes public keys along and cannot work the secret out.
2. **Mask.** Each round, both phones turn the secret into the same list of random numbers. Phone A adds them,
   phone B subtracts them.
3. **Add.** The server adds the two uploads. The random numbers cancel and the real sum is left: the FedAvg result.
4. **Compress (Shiv's part).** Instead of its full weights, each phone sends only this round's change, already
   multiplied by its share of the data, rounded to 8 bits on a scale the server announces. The two rounded values
   always add up to something that fits in 16 bits, so a 16-bit mask is enough: 2 bytes per number instead of 16
   hex digits.
5. **Local models.** The local-only models are encrypted with the secret before the server relays them.
6. **Dropout.** One masked upload on its own is noise, so if a phone is missing the server throws the round away and
   repeats it with a new mask; the global model stays as it was. If a phone restarts, both phones agree on a new secret.

### Where it is in the code

| Step | Where |
|---|---|
| Key agreement, masking, encryption on the phone | `core/SecureAggregation.kt` → `sharedKey`, `maskUpdate`, `maskCompressed`, `encrypt`; class `SecureSession` |
| When the phone does it | `core/FlClient.kt` → `agreeOnKeys`, and the secure branches of `federatedRound` |
| Adding the masked uploads | `Sources/server/secure_agg.py` → `sum_masked`, `sum_compressed`; `fl_server.py` → `aggregate` |
| The scale for compression | `fl_server.py` → `aggregate` (next `clip` = 1.5 × largest reported change) |
| Discarding a round | `fl_server.py` → `discard_round` (called from `tick` on timeout) |
| Refusing plain weights in this mode | `fl_server.py` → `update`, `put_local_model` |

### Demo (after the main presentation)

1. SSH window: `bash start_server.sh PASSWORD --secure --compress --round-timeout 10` (or tick **secure aggregation**
   and **+ compression** on the dashboard's *Start a new run*); **Start** on both phones.
2. Dashboard → *What the phones upload* → raw JSON: the upload is now a 208-character string of masked numbers.
3. Results: the global model is as accurate as before; the row *One upload, read by the server* is at guessing level,
   and the *Curious server* card drops from 6 / 6 (plain run) to about 3 / 6.
4. Dashboard section 5 *Compare runs*: the plain run and this run side by side (accuracy, curious-server score,
   upload size, phone cost).
5. Put one phone in **Airplane mode** (with the cloud server, turning off Wi-Fi alone lets the phone continue over
   mobile data): the dashboard shows rounds discarded. Airplane mode off: the phone reconnects and training continues.

### Experiments for the report

Run `bash start_server.sh PASSWORD --sweep secure_study` (plain, secure and compressed secure, 3 seeds each; both
phones tick *Keep following new server runs*), download the runs, then `python3 analyze_run.py --latest 9`.

| Question | Simulated phones (synthetic data) | Real phones |
|---|---|---|
| Is accuracy unchanged? | global 100% in all three modes | to measure |
| What can the server read from one upload? | plain: a 100%-accurate model; secure: 19.4% ± 3.9; compressed: 25.3% ± 11.6 (guessing is 16.7%) | to measure |
| Upload size per round | plain 1,886 B; secure 1,767 B; compressed secure 490 B | to measure |
| Dropout | plain averages with one phone; both secure modes discard and repeat the round, then recover | to measure |

### Limits to say out loud

* With only two phones, each phone can work out the other's update from the new global model. The protection
  is against the server.
* The server is assumed to relay the public keys honestly; HTTPS (Caddy) stops anyone in between from swapping them.
* The total sample count, the loss and (with compression) the largest change are still sent unmasked. Per-pose counts
  are not: we found they revealed each phone's poses, so secure mode now leaves them out and the server refuses them.
* Compression rounds each change to 255 levels; on synthetic data that cost no accuracy.

## 7. More Q&A answers (from Shiv's FedPose notes, adapted to the combined app)

**Why does federated beat local?** Each phone only saw three of the six poses, so its local model has never been
told the other three exist. Their scores are only ever pushed down during training, so it gets them wrong every time
and tops out near 50%. FedAvg combines the two phones' models, so the global model learns all six without either
phone sharing data.

**Why are mean gravity features enough?** A still pose is defined by which axis of the phone gravity points along:
plus or minus x, y or z. The six poses sit in different places of that 3D gravity vector, so a linear (softmax)
model can separate them. The mean over 1 second removes hand jitter; the standard deviation says how steady the
phone is.

**Why would the gyroscope not help?** It measures rotation speed, not orientation. In a still pose it reads about
zero for every pose, so it adds noise rather than information. Check whether our sensor study confirms this before
saying it.

**Why does a local model get unseen poses wrong instead of being unsure?** It never sees those labels, so their
weights and biases only ever decrease, and it confidently picks one of its own three poses.

**What does the server see?** Model weights (or masked numbers), sample counts, per-pose counts (plain mode only) and
scores. Never sensor values. The dashboard's upload section proves it, and the server refuses any field not on its list.

**What if one phone drops out?** In normal mode the round times out and the server averages what it has; a late
upload is rejected as stale and the phone re-syncs. During the outage the global model drifts toward the phone that
is still training, and recovers when the other returns. In secure mode the round is repeated instead.

**Does compression help?** It cuts each upload to about a quarter, but in absolute terms the saving is small because
the model has only 78 numbers. Its real value is that secure aggregation no longer makes uploads bigger, and it would
matter more for larger models.

**Why did you build two versions?** We each built the full pipeline to check each other's design, then combined the
stronger parts: Hasin's app, server and cloud setup with Shiv's multi-window blind test, 8-bit compression and Q&A notes.
