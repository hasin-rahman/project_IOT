# Demo-day guide: Federated Phone-Pose Recognition

CSC 8223 · Hasin Rahman and Shiv Brahmbhatt · written Nov 13, 2026 from the project handout, our proposal and the
demo-day slides in `Presentation guidlines/`.

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
| Blind test | `app/PredictActivity.kt` → `startCapture`, `captureWhenStill`, `capture` | Tap → 3 beeps → wait until still for 1 s → record one window → show and log the three answers. |
| Practice score | `core/Live.kt` → `LiveTest.rubricPoints` | 30 × global accuracy + 5 × local A on A's poses + 5 × local B on B's poses. |
| Dropout recovery | `fl_server.py` → `tick`; `FlClient.kt` → `run` (the `IOException` branch and the 409 case) | The server closes a round after 30 s with whatever arrived. A phone that was offline gets "stale", downloads the newest model and continues. |

## 4. Core findings to state

Fill in the numbers after the real recordings (none exist yet; everything so far ran on synthetic test data).

1. **Federation gives what local training cannot.** Each local model only knows three poses, so on a six-pose test it
   is right about half the time (about 100% on its own poses, about 0% on the others). The global model has seen
   neither phone's data, yet recognises all six. On synthetic data: local 50%, global 100%.
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
| Server: rounds, FedAvg, storing and relaying models, privacy allow-list and upload log | Hasin |
| Sensor/feature variations (how the feature sets are built) | Hasin |
| On-device model and training (softmax, SGD), weight upload and download | Shiv |
| Evaluation: metrics, three-model comparison, analysis script and plots | Shiv |
| Sensor study results and what they mean | both |

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

## 6. Extra credit: secure aggregation with dropout handling (+3 to 5 minutes)

We replaced the three small extras in the proposal (8-bit compression, adaptive participation, dropout recovery)
with one larger extension, because the handout says several small add-ons do not earn more than one substantial,
well-evaluated one, and because compressing a 78-number model saves almost nothing. Dropout recovery is kept and
is part of this extension. The report should state this change.

### What it does, in plain words

Normally the server sees each phone's weights. With secure aggregation it sees only scrambled numbers, and still
gets exactly the same average.

1. **Shared secret.** Each phone makes a private number and sends the matching public key through the server.
   Each phone combines its private number with the other's public key and both arrive at the same secret
   (Diffie-Hellman). The server only passes public keys along and cannot work the secret out.
2. **Mask.** Each round, both phones turn the secret into the same list of random numbers. Phone A adds them to
   its weights, phone B subtracts them.
3. **Add.** The server adds the two uploads. The random numbers cancel and the real sum is left, which it
   divides by the total sample count: the FedAvg result.
4. **Local models.** The local-only models are encrypted with the secret before the server relays them.
5. **Dropout.** One masked upload on its own is noise, so if a phone is missing the server throws the round
   away and repeats it; the global model stays as it was.

### Where it is in the code

| Step | Where |
|---|---|
| Key agreement, masking, encryption on the phone | `core/SecureAggregation.kt` → `sharedKey`, `maskUpdate`, `encrypt`; class `SecureSession` |
| When the phone does it | `core/FlClient.kt` → `agreeOnKeys`, and the `masked_model` branch of `federatedRound` |
| Adding the masked uploads | `Sources/server/secure_agg.py` → `sum_masked`; `fl_server.py` → `aggregate` |
| Discarding a round | `fl_server.py` → `discard_round` (called from `tick` on timeout) |
| Refusing plain weights in this mode | `fl_server.py` → `update`, `put_local_model` |

### Demo (after the main presentation)

1. Dashboard: tick **secure aggregation** in *Start a new run*, start it; **Start** on both phones.
2. Dashboard → *What the phones upload* → raw JSON: the weights are now 78 hex numbers.
3. Results: the global model is as accurate as before; the row *One upload, as the server sees it* is at
   guessing level (in the normal mode it is a working model).
4. Switch one phone's Wi-Fi off: the dashboard shows the round discarded. Switch it back on: training continues.

### Experiments for the report

Run `python3 fl_server.py --sweep secure_study` (plain and secure, 3 seeds each, both phones following runs), then
`python3 analyze_run.py --latest 6`. Fill in from real phone runs:

| Question | Simulated phones (synthetic data) | Real phones |
|---|---|---|
| Is accuracy unchanged? | global 100% in both modes | to measure |
| What can the server read from one upload? | plain: 100% accurate model; secure: 9.7% ± 7.1 (guessing is 16.7%) | to measure |
| Cost | update 1,886 B plain, 1,767 B masked; no measurable extra time per round | to measure |
| Dropout | plain averages with one phone; secure discards and repeats the round, then recovers | to measure |

### Limits to say out loud

* With only two phones, each phone can work out the other's update from the new global model. The protection
  is against the server.
* The server is assumed to relay the public keys honestly.
* Sample counts, per-pose counts and the loss are still sent unmasked.

### Who answers

* Key agreement, masking on the phone (`SecureAggregation.kt`, `FlClient.kt`): Hasin.
* Adding and discarding on the server, and the comparison results: Shiv.
(Adjust to however you two actually split it.)

