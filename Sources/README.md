# Federated Phone-Pose Recognition — Sources

CSC 8223 course project · Hasin Rahman and Shiv Brahmbhatt

Two Android phones each record their own motion-sensor data, train a small classifier on it, and
share **only model weights** with a Python server that combines them with FedAvg. Raw sensor data
and extracted features never leave the phones.

> **Status: version 0.7.** Recording, on-device training, federated learning with dropout recovery, the live
> three-model screen with a **blind-test mode** (the judges choose the pose) and a practice mode scored like the demo
> rubric, **upload evidence** for the privacy rule on the server and on the phone, experiment sweeps for the sensor
> study, the analysis script for the report, and the server on **Google Cloud with HTTPS** (§13).
> Extra credit (§12): **secure aggregation** and **compressed secure aggregation**, evaluated with a
> **curious-server attack** (what the server can learn from one upload) and the **cost on the phones** (time, data,
> battery). **Pose actions** (§6) close the IoT loop: the phone acts on the pose it recognises.
> The dashboard has a **Compare runs** table for plain, secure and compressed runs side by side.

**Who built what.** Two complete versions of the project were built in parallel and then combined into this one:
Hasin's (`main`: this app, server, dashboard, secure aggregation, cloud setup) and Shiv's FedPose version
(branch `fedpose`: its own app, Flask server, Python reference with experiment scripts, 8-bit update compression,
adaptive participation, a multi-window blind test and the demo-day Q&A notes). The combined app keeps Hasin's code
as the base and adds from Shiv's: the multi-window blind-test answer (§6), the 8-bit compression inside compressed
secure aggregation (§12) and the demo-day answers (`Demo_Day_Guide.md`). Both phones now record with this app.
The `fedpose` branch stays in the repository as the record of Shiv's version.

**Quick start:** start the server (§1), install the app on both phones (§2), set them up (§3), record (§4),
press **Start** on both phones' *Federated training* screen (§5), then open *Live demo and blind test* (§6).

## Folder layout

```
Sources/
├── README.md                  this file
├── android/                   Android Studio project (Kotlin, no third-party libraries)
│   └── app/src/main/java/com/posefl/
│       ├── core/              pure Kotlin (no Android imports, unit-tested on a laptop):
│       │                        poses, windowing + features, softmax model, metrics,
│       │                        FlClient.kt (the federated client), Live.kt (live prediction, blind test, practice score),
│       │                        SecureAggregation.kt (extra credit), DeviceCost.kt (time, data, battery per run),
│       │                        PoseActions.kt (when an action switches on and off), Json.kt
│       └── app/               Android screens: Main, Record, Sessions, Train (federated), Predict (live demo + blind
│                                test), PoseActions (the phone acts on the recognised pose)
├── server/
│   ├── fl_server.py           federated server (Python standard library only)
│   ├── fedavg.py              sample-weighted Federated Averaging
│   ├── secure_agg.py          secure aggregation: key agreement, masking, unmasking by adding (extra credit)
│   ├── curious_server.py      the curious-server attack: which poses can the server guess from one upload?
│   ├── run_summary.py         one line of numbers per run (the dashboard's Compare runs table)
│   ├── dashboard.html         the dashboard page the server shows at /
│   ├── config.json            experiment settings (feature set, rounds, learning rate, ...)
│   ├── analyze_run.py         run folders -> report tables (summary.md, results.csv) and figures
│   ├── sim_clients.py         two simulated phones (synthetic data) for testing the server
│   └── test_server.py         server tests (rounds, stale updates, dropout timeout, sweeps)
├── cloud/                     Google Cloud with HTTPS (§13): setup_caddy.sh, start_server.sh, stop_server.sh
└── reference/
    ├── posefl_ref/            Python mirror of the phone pipeline (features, model, metrics, synthetic data)
    ├── simulate.py            offline local-vs-federated simulation on synthetic data
    ├── crosscheck/            checks that the Kotlin code computes exactly what the Python reference computes
    └── e2e/                   runs the app's Kotlin federated client as two phones against the real server
```

## Requirements

| Component | Version used / required |
|---|---|
| Server | Python 3.9 or newer (tested with 3.13). No packages needed. |
| Reference, simulator and tests | Python 3.9+ with `numpy` (tested with numpy 2.5) |
| Analysis figures (optional) | `matplotlib` (`pip3 install matplotlib`); without it the tables are still written |
| Android build | Android Studio (recent version) with its bundled JDK 17+. The project uses Android Gradle Plugin 8.7.3, Gradle 8.11.1, Kotlin 2.0.21 and compileSdk 35. |
| Phones | Android 8.0 (API 26) or newer with an accelerometer. Set up for the Samsung Galaxy S24 (Android 14 or newer); targetSdk is 34. |
| Network | Server on the Mac: phones and Mac on the same Wi-Fi. Server on Google Cloud (§13): any internet connection. |

## 1. Start the server (on the Mac)

```bash
cd Sources/server
python3 fl_server.py
```

The server prints the URL to enter on the phones, for example `http://192.168.1.23:8000`. To see live
status (connected clients, rounds and traffic), open http://localhost:8000/ in a browser.

* If macOS asks whether Python may accept incoming connections, click **Allow**.
* Settings live in `config.json`. Command-line flags override them, for example
  `python3 fl_server.py --feature-set acc+grav+gyro --rounds 30`.
* Each run writes to `server/runs/<run_id>/`: the global model after every round, `rounds.jsonl`,
  `metrics.jsonl`, the local-only models, `traffic.json` (bytes per client and endpoint) and
  `uploads.jsonl` (every request body the phones sent, exactly as received).

## 2. Build and install the app (Android Studio)

1. In Android Studio, choose **File → Open** and select the `Sources/android` folder.
   The first Gradle sync downloads the build tools, which needs internet and takes a few minutes.
   If Android Studio offers to upgrade the Android Gradle Plugin, you can accept or skip.
2. On each phone, turn on developer mode. On a Samsung: **Settings → About phone → Software information**,
   then tap **Build number** 7 times. Then go to **Settings → Developer options** and turn on **USB debugging**.
3. Connect the phone by USB, accept the debugging prompt, select the phone in the device menu and press **Run ▶**.
   On a Galaxy S24, if the phone does not appear or the install is refused, turn off
   **Settings → Security and privacy → Auto Blocker** (it blocks installs over USB), then reconnect.
4. (Optional) To run the JVM unit tests, right-click `app/src/test/.../CoreTest.kt` and choose **Run 'CoreTest'**.

## 3. Set up each phone

On the app's main screen:

1. Choose **Client A** on one phone and **Client B** on the other.
2. Enter a **Participant ID**: the name of whoever is holding the phone. Change it when a different person records.
3. Enter the **Server URL** that the server printed and tap **Save and test connection**. It should say *Connected*.
   Leave **Server password** empty unless the server was started with `--password` (for example on Google Cloud, §13).
   The bottom of the screen lists which sensors the phone has.

## 4. Record data

Open **Record data**, then:

1. Pick the **split**:
   * **Train** — only this client's three poses (A: Screen Up, Screen Down, Portrait Up; B: Portrait Down,
     Landscape Left, Landscape Right). This creates the required non-IID split.
   * **Validation** or **Test** — any of the six poses. These sessions are used for tuning and evaluation only, never for training.
2. Pick the pose, how the phone is held, and the duration. 20 s gives about 35 usable windows.
3. Press **Start**. Three short beeps give you time to get into position. A long beep starts the recording and a
   double beep ends it. The beeps and vibration let you record Screen Down without seeing the screen.
   The first and last second are discarded automatically.
4. The panel labeled **Phone looks like** is a live rule-based check of the gravity direction. It turns
   green when the phone matches the selected pose. After each recording, the app shows the share of samples that
   matched the label, warns below 80%, and asks whether to **Keep** or **Discard** the session.

**Recommended amounts per phone**

| Split | Poses | Sessions per pose |
|---|---|---|
| Train | the client's 3 poses | 6 or more × 20 s, spread over 2–3 people, in hand and on a surface, with natural tilts up to about 30° |
| Validation | all 6 | 1–2 × 20 s |
| Test | all 6 | 2 × 20 s, plus a separate set from a person who gave no training data |

**Recorded sessions** shows the usable windows per pose and split. Tap a session for details; long-press to delete it.

**Privacy:** recordings are stored in the app's private storage on the phone. The app has no export or upload
for sensor data. Only model parameters, sample counts and evaluation metrics are sent to the server.

## 5. Federated training (both phones)

1. Start the server on the Mac (§1) and open http://localhost:8000/ to watch.
2. On **each** phone open **Federated training** and press **Start**. Order does not matter.

Each phone then, automatically:

1. builds features from its own **train** sessions (only its three poses) and trains the **local-only model from scratch**
   (`baseline_epochs`), uploads it so the other phone can compare against it later;
2. joins the federated rounds: download the global model → measure it on its validation sessions → train a copy for
   `local_epochs` → upload. The server averages when both updates are in;
3. after the last round, downloads the other phone's local model and evaluates **all three models** on its validation,
   test and *unseen-user* test sessions (test sessions from a participant who recorded no training data on that phone),
   uploads the metrics, and saves the three models on the phone for the live screen.

The phone screen and the dashboard show the round, the global model's validation accuracy per round, and the final
comparison table. Training takes seconds; most of the time is the two phones waiting for each other.

**What is uploaded (privacy evidence).** A phone sends only: the model weights `W` and `b`, the number of training
windows, the number of windows per pose (`class_counts`, plain mode only: secure mode hides it, §12), the training
loss/accuracy and timing, and at the end its accuracy scores and what the run cost on the phone (time, data, battery).
You can show this three ways:

* the dashboard section **What the phones upload** shows the latest request of each kind from each phone, with the raw JSON;
* the server console prints every upload (`upload from A: POST /update, 1948 bytes: ... model{W[72], b[6]} ...`),
  and `runs/<run_id>/uploads.jsonl` keeps all of them;
* on the phone, **Show what this phone uploaded** on the training screen shows the exact JSON it sent.

The server also enforces the rule: `ALLOWED_UPLOAD_FIELDS` in `fl_server.py` lists the only fields a phone may send.
A request with any other field (sensor rows, feature vectors, ...) is refused with HTTP 400.

**Independent baselines.** Each local-only model starts from all-zero weights and is trained only on its own phone's
three poses (`trainLocalOnlyModel` in `core/FlClient.kt`); it is never initialised from or averaged with the global
model. The results on the phone and the dashboard's **Recall per pose** table show it at about 0% on the other
client's poses.

**Dropout recovery (our extension).** If a phone loses Wi-Fi, the other one keeps going: once a round has started
the server waits at most `round_timeout_s` (30 s) and then averages what it has, recording the missing phone. The
disconnected phone retries every 2 s; when it is back, its late update is rejected as stale (HTTP 409), it downloads
the newest global model and continues. To demonstrate: turn off Wi-Fi on phone B during training, watch the dashboard
log *"timeout … no update from ['B']"*, turn Wi-Fi back on.

## 6. Live screen, blind test and practice

Open **Live demo and blind test** on either phone (after training). The top card shows, five times a second, what
the **federated global model** and the two **local-only** models predict for the same 1-second sensor window. It
uses the three models saved on the phone, so it works without the server.

**Blind test (demo day; selected by default).** The judges choose the pose from their pose cards; the app does not
know it.

1. Tap **Capture placement**. Three short beeps give time to put the phone in the pose.
2. Keep the phone still. As soon as it has been still for 1 second (at most 8 s after the beeps) the app takes
   each model's answer, gives a long beep and a vibration, and shows what each of the three models predicted.
   The answer is **averaged over the last two 1-second windows** (the newest one and the one ending 0.5 s
   earlier), so one bump in one window cannot decide it (`LivePredictor.captureAnswers`; the multi-window idea
   comes from Shiv's version).
3. The result stays on the screen until the next capture, so a Screen Down placement can be read after picking the
   phone up. Every placement is numbered (1 to 36) and listed below the buttons. **Undo last** removes a mistake;
   **Finish / save** stores the list on the phone (`live_tests/blind_test_*.csv` in app storage).

**Practice (for rehearsals).** Select *Practice*: the app asks for a pose (each of the six once per block, in random
order, so 36 trials = every pose 6 times), captures the same way, marks each model's answer right or wrong, and
keeps the score the way part A of the demo rubric does: 30 × global accuracy on all placements + 5 × local A accuracy
on A's poses + 5 × local B accuracy on B's poses. It also counts how often a local model got one of the *other*
phone's poses right (this should be about 0). **Finish / save** stores the trials on the phone and sends the scores,
not sensor data, to the server as test set `live_test`.

**Pose actions (the IoT loop: sense → decide → act).** Open **Pose actions**. The phone acts on the pose the chosen
model recognises: when the model has seen the same pose for **3 seconds** with at least 75% confidence, its action
switches on; when the phone has been out of that pose for **1 second**, it switches off.

| Pose | Client | Action |
|---|---|---|
| Screen Down | A | Do Not Disturb on (the app asks once for *Do Not Disturb access* in Settings) |
| Screen Up | A | Desk clock: a big clock fills the screen |
| Portrait Down | B | Flashlight on |
| Landscape Left | B | A timer starts on the screen |
| Portrait Up, Landscape Right | | no action |

The switch at the top chooses which model decides: **Global**, **Local A** or **Local B**. A local model only knows
its own phone's three poses, so it can switch on only two of the four actions; the global model can switch on all
four. For example, on phone A with *Local A*, turning the phone upside down does nothing (or the wrong thing),
and with *Global* the flashlight turns on: phone B taught the pose, and its data never left phone B.
Everything switches off when you leave the screen. Code: `core/PoseActions.kt` (`PoseActionTrigger`, unit-tested)
and `app/PoseActionsActivity.kt` (`switchOn`, `switchOff`).

## 7. Experiments for the report

All runs are written to `server/runs/<run_id>/`. On both phones tick **Keep following new server runs** before
pressing Start: the phones then train again whenever the server starts a new run.

| What | Server command |
|---|---|
| Baseline, 3 seeds (mean ± spread) | `python3 fl_server.py --sweep seeds` |
| **Sensor study** (required extension): `acc`, `acc+grav`, `acc+grav+gyro`, `acc+grav+gyro+rot` × 3 seeds = 12 runs | `python3 fl_server.py --sweep sensor_study` |
| A single run with other settings | `python3 fl_server.py --feature-set acc+grav+gyro --rounds 30 --seed 1` or the *Start a new run* form on the dashboard |
| Your own list of runs | `python3 fl_server.py --sweep my_sweep.json` (a JSON list of overrides, e.g. `[{"local_epochs": 5}, {"lr": 0.5}]`) |
| Dropout | `python3 fl_server.py --round-timeout 10`, then switch a phone's Wi-Fi off and on during training |
| **Secure aggregation** (extra credit): plain FedAvg, secure aggregation and compressed secure aggregation × 3 seeds = 9 runs | `python3 fl_server.py --sweep secure_study` |
| Dropout with secure aggregation | `python3 fl_server.py --secure --round-timeout 10`, then switch a phone's Wi-Fi off and on |

A sweep starts the next run as soon as both phones have reported their results, and prints the analysis command at the end.

## 8. Analysis: tables and figures for the report

```bash
cd Sources/server
python3 analyze_run.py runs/run_20261120_141502          # one run
python3 analyze_run.py --latest 12                        # e.g. the 12 sensor-study runs
```

It writes `analysis/<timestamp>/` with `summary.md` (three-model comparison with accuracy and macro-F1 per test set,
per-class recall, the sensor-study table, rounds with a missing phone, round of convergence and bytes per update),
`results.csv`, and the figures `accuracy_by_round.png`, `model_comparison.png`, `confusion_<set>.png` and
`sensor_study.png`. With several runs of the same feature set, values are mean ± standard deviation over the runs.

**Settings (`server/config.json`)**

| Key | Default | Meaning |
|---|---|---|
| `feature_set` | `acc+grav` | input features (§9) |
| `rounds` | 20 | federated rounds per run |
| `local_epochs` | 1 | epochs each phone trains per round |
| `baseline_epochs` | 20 | epochs for the local-only models (trained from scratch) |
| `lr`, `batch_size`, `l2` | 0.1, 16, 1e-4 | SGD settings, used by both phones |
| `seed` | 0 | shuffling seed on the phones |
| `round_timeout_s` | 30 | after the first update of a round arrives, wait this long for the other phone (0 = forever) |
| `min_clients` | 1 | updates needed to close a round on timeout |
| `secure_aggregation` | false | phones upload masked weights; the server only learns their average (§12) |
| `compression` | false | with secure aggregation: 8-bit changes under 16-bit masks, 2 bytes per parameter (§12) |
| `quant_clip` | 4.0 | compression: the scale of the first round (later rounds follow the reported changes) |

## 9. Pipeline specification (shared by the Kotlin and Python code)

* **Raw row** (about 50 Hz, one row per accelerometer event, with the latest value of every other sensor):
  `t_ms, ax, ay, az, gx, gy, gz, wx, wy, wz, rx, ry, rz, rw`.
  These are the accelerometer, gravity, gyroscope and rotation-vector quaternion.
* **Windows:** 1 s long, starting every 0.5 s (50% overlap). The first and last 1 s of each session are dropped, and a window needs at least 25 samples.
  Data is split into train/validation/test **by session, before windowing**, so overlapping windows never cross splits.
* **Features:** a feature set joins blocks with `+`. Each block gives the mean and the population standard deviation of x, y and z (6 numbers).
  * `acc`: accelerometer ÷ 9.81
  * `grav`: gravity sensor ÷ 9.81
  * `gyro`: gyroscope in rad/s
  * `rot`: the device-frame "up" vector computed from the rotation vector. Compass heading is removed, so it does not depend on which way the user faces.
  * The baseline is `acc+grav` (12 features). The sensor study compares `acc`, `acc+grav`, `acc+grav+gyro` and `acc+grav+gyro+rot`.
* **Model:** softmax regression with weights W (6 × F) and biases b (6); the baseline has 78 parameters. It is trained with mini-batch SGD
  (batch 16, learning rate 0.1, L2 weight 1e-4).
* **FedAvg:** the server averages the client models' parameters, weighted by each client's number of training windows.
* **Live prediction:** the same 1 s window, recomputed every 200 ms; each model's class probabilities are averaged over
  the last 4 windows. Below 40% confidence the live answer is shown in orange (chance is 17%). A captured test placement
  averages each model's probabilities over the newest window and the one ending 0.5 s earlier, and takes the top class.
  The phone counts as moving when the standard deviation of the acceleration magnitude in the window exceeds 0.06 g.

## 10. Tests and the offline simulation

```bash
cd Sources/server && python3 -m unittest -v test_server
python3 sim_clients.py                  # with fl_server.py running: two simulated phones over HTTP
cd ../reference && python3 simulate.py  # offline local-vs-federated comparison (synthetic data)
```

The server tests (20) cover rounds, stale updates, the privacy allow-list, dropout, sweeps, the password, secure
and compressed aggregation, the refusal of per-pose counts in secure mode, the curious-server attack (6/6 in plain
mode, about chance with secure aggregation), the device-cost upload and the Compare runs summary. The Kotlin JUnit
tests (`android/app/src/test/.../CoreTest.kt`, 21) cover features, the model, the federated client, the blind test,
secure aggregation, compression, the pose-action trigger (3 s on, 1 s off) and the cost tracker.

**Kotlin client end to end.** `reference/e2e` runs the app's own `core/` Kotlin code (federated client, live
predictor, test mode) as two phones against the real server, including a simulated Wi-Fi loss on phone B and a sweep.
Commands are at the top of `make_sessions.py`; it needs `kotlinc` (or run `CoreTest` in Android Studio, which covers
the same core code without a server).

**Kotlin–Python cross-check.** `reference/crosscheck` writes six synthetic sessions, runs them through both the
Kotlin and the Python pipelines, and compares features, trained weights and metrics. They agree to about 1e-16. The
commands are at the top of `crosscheck.py`. Compiling the Kotlin side needs `kotlinc`, or you can use the JUnit tests in Android Studio instead.

The synthetic data is a simple physical model used only to test the code (it is so clean that every feature set
scores 100%). All results reported in the project come from real phone recordings.

## 11. Demo-day checklist

* Before presenting: server running, both phones on the same network, **Federated training** finished on both
  phones (all three models loaded on the live screen), media volume up for the beeps.
* To retrain in front of the audience: record a short train session on each phone, click **Start a new run** on the
  dashboard, then press **Start** on both phones (press **Stop** first if a phone is still following runs, so it
  reloads the new recording).
* The models used by the live screen are those of the **last finished run**, so end on the feature set you want
  for the blind test.

* Network: the Google Cloud server (§13) works over campus Wi-Fi or mobile data. Test it in the classroom beforehand.
  As a backup, bring a phone hotspot for the Mac and both phones (campus Wi-Fi often blocks phone-to-laptop
  connections), with the server ready to start on the Mac.
* Train the night before, so the models are already saved on both phones; the live screen works with no server.
* Charge both phones, keep the APK on a third phone, and record a backup video of a full run.
* Rehearse test mode with people who have never used the app.
* For battery numbers in *Cost on the phones*, unplug the phones during the run (a charging phone reports nothing).
* Pose actions: grant *Do Not Disturb access* once beforehand (the button on the Pose actions screen).

## 12. Secure aggregation (extra credit)

In the normal mode the server receives each phone's weights. With secure aggregation switched on, the server
receives only masked numbers and still computes exactly the FedAvg average. The code is
`android/.../core/SecureAggregation.kt` on the phones and `server/secure_agg.py` on the server.

**How it works**

1. **Key agreement (Diffie-Hellman).** At the start of a run each phone creates a private number and publishes
   the matching public key through the server (`POST /pubkey`). Each phone combines its own private number with
   the other phone's public key; both get the same shared key. The server only relays public keys and cannot
   compute it.
2. **Masking.** Every round, both phones turn the shared key into the same list of random 64-bit numbers. Each
   phone converts `num_samples × weight` to a whole number; phone A adds the mask and phone B subtracts it
   (modulo 2^64). The upload field is `masked_model` instead of `model`.
3. **Unmasking by adding.** The server adds the two uploads. The masks cancel, leaving
   `n_A·w_A + n_B·w_B`, which it divides by `n_A + n_B`.
4. **Local-only models.** They are encrypted with the shared key before upload, so the server relays them to
   the other phone without being able to read them.

In this mode the server refuses plain weights (`model`) in `/update` and `/local_model`.

**Dropout.** One masked update alone is noise, so a round with a missing phone cannot be used. After
`round_timeout_s` the server discards the round and it is repeated with a new mask; the global model stays as it
was. (Plain FedAvg instead averages whatever arrived.) If a phone restarts mid-run it publishes a new key and both
phones compute a new shared key automatically.

**Compressed secure aggregation** (`--secure --compress`; Hasin's masking combined with Shiv's 8-bit compression).
Masking with 64-bit numbers makes each upload large. In compressed mode each phone instead sends the *change* of its
weights this round, multiplied by its share of all training windows (`n_i / total`), clipped to ±`clip` and rounded
to one of 255 levels (8 bits). The server announces `clip` and `total_samples` with each round, so both phones use the
same scale. The two weighted values add up to at most ±127 levels, so 16-bit masks are enough: **2 bytes per
parameter** (the field is `masked_delta`). The server adds the uploads modulo 2^16, the masks cancel, and
`sum × clip / 127` is the FedAvg change, accurate to one rounding step. Each phone also reports the largest change it
made (one number, in the clear), and the next round's `clip` is 1.5 × the largest reported change, so the scale
follows training as the changes get smaller. Values beyond `clip` are clipped and counted (`clipped`).
Code: `maskCompressed` in `SecureAggregation.kt` / `mask_compressed` in `secure_agg.py`, and `sum_compressed` on the server.

Results with two simulated phones (synthetic data, 3 seeds per mode; to be measured again on the real phones):

| mode | global accuracy | one upload as the server sees it | update size |
|---|---|---|---|
| plain FedAvg | 100% | 100% (a working model) | 1,886 B |
| secure aggregation | 100% | 19.4% ± 3.9 (guessing is 16.7%) | 1,767 B |
| compressed secure aggregation | 100% | 25.3% ± 11.6 | 490 B |

**How to run it**

* One run: `python3 fl_server.py --secure` (add `--compress` for the compressed mode), or tick **secure aggregation**
  (and **+ compression**) in the dashboard's *Start a new run* form. On Google Cloud: `bash start_server.sh PASSWORD --secure --compress`.
* The comparison for the report: `python3 fl_server.py --sweep secure_study` with **Keep following new server runs**
  ticked on both phones, then `python3 analyze_run.py --latest 9`. The summary gets a table *Secure aggregation
  compared with plain FedAvg*: global accuracy, "one upload as the server sees it", update size, time per round and
  discarded rounds, for all three modes.

**What to look at**

* Dashboard → *What the phones upload* → raw JSON of `/update`: 78 hex numbers instead of weights
  (compressed mode: a 208-character base64 string, `masked_delta`).
* Phone → **Show what this phone uploaded**: the same masked numbers.
* Results on the phone and the dashboard have an extra row `server_sees`: each phone scores its own last upload the
  way the server received it. In plain FedAvg that is a working model; with secure aggregation it is at guessing level.

**Curious-server attack: what can the server learn from one upload?** (`server/curious_server.py`)

The server is assumed to be honest but curious: it runs federated learning correctly but looks closely at what it
receives. Can it work out **which poses a phone has recorded**? In a round, a phone starts from the global model the
server sent and trains on its own data, which raises the bias of every pose it has examples of and lowers the others.
The server knows the model it sent, so it subtracts it from the upload and guesses "the poses whose bias went up".
Each upload gets a score out of 6 (random guessing: 3).

* Plain FedAvg: **6/6 on every upload** (it names each phone's three poses every round). The per-pose counts
  (`class_counts`) gave the answer away directly too.
* Secure aggregation: the upload is masked, so the guess is a coin flip (about 3/6). We found that the phones still
  sent `class_counts` in this mode, which would have revealed the poses without any attack; in version 0.7 the phones
  leave it out in secure mode and the server refuses it.

`python3 curious_server.py runs/<run_id>` prints every round's guess and score; the dashboard shows it live
(card *Curious server* and the *Privacy* section) and in *Compare runs*.

**Cost on the phones** (`core/DeviceCost.kt`). At the end of each run a phone reports what the run cost it: training
time per round, the local-only model, building the features of one window, classifying one window with the three
models, bytes sent and received, the whole run's time and the battery used (from the battery's charge counter;
nothing is measured while the phone is charging). It is shown on the phone (*Cost on this phone*), sent with
`POST /device_cost` (timings and battery only, on the allow-list) and shown on the dashboard.

**Compare runs.** The dashboard's last table has one row per run: mode, global and local accuracy, what the server
reads from one upload, the curious-server score, upload size per round, train time per round and battery. Run a plain
run and a secure + compressed run with the same recordings and they sit next to each other. For the report,
`analyze_run.py` writes the same comparison (section *Curious server: what one upload reveals, and what it costs on
the phone*).

**Limits**

* With two clients, each phone can work out the other's update from the new global model. The protection is
  against the server, not against the other client.
* The server is assumed to follow the protocol and relay the public keys honestly.
* The total number of training windows and the training loss are still sent in the clear (with compression also
  the largest change per round, one number). The per-pose counts are not: in secure mode the phones leave out
  `class_counts` and the server refuses it.
* Compression rounds each weighted change to 255 levels; with the synthetic data this did not change the accuracy.

## 13. Running the server on Google Cloud with HTTPS (Caddy)

With the server on a Google Cloud machine, the phones only need internet access (any Wi-Fi or mobile data), so
the two phones can train from different places and the Mac can be off. **Caddy** runs on the same machine, gets a
free HTTPS certificate and passes requests on to the server, so all traffic, including the password, is encrypted.
HTTPS also protects the secure-aggregation key exchange: nobody in between can swap the phones' public keys.

```
phones --HTTPS--> Caddy (ports 80/443) --> fl_server.py (127.0.0.1:8000, password required)
```

The server address is `https://34-1-2-3.sslip.io` for the external IP `34.1.2.3`; the free service sslip.io turns
that name back into the IP, which is what lets Caddy get a certificate without buying a domain.

### One-time setup (about 30 minutes)

1. **Account.** Sign up at https://cloud.google.com/free (a card is required). The free tier includes one e2-micro
   machine per month in `us-west1`, `us-central1` or `us-east1`. Check the current limits there, and set a budget
   alert under **Billing → Budgets & alerts** (for example $1).
2. **Create the machine.** Console → **Compute Engine → VM instances → Create instance** (enable the Compute Engine
   API if asked):
   * Name `posefl-server`, region **us-east1**, machine type **e2-micro**.
   * Boot disk: **Ubuntu 24.04 LTS**, standard persistent disk, 10 GB.
   * Firewall: tick **Allow HTTP traffic** and **Allow HTTPS traffic** (Caddy needs both).
   * Click **Create**, and note the **External IP** in the VM list.
3. **Upload the files.** Click **SSH** next to the machine. In the window that opens, click **Upload file** and
   upload these 10 files (from `Sources/server` and `Sources/cloud`):
   `fl_server.py`, `fedavg.py`, `secure_agg.py`, `curious_server.py`, `run_summary.py`, `dashboard.html`,
   `config.json`, `setup_caddy.sh`, `start_server.sh`, `stop_server.sh`. Then put them in one folder:
   ```bash
   mkdir -p posefl && mv fl_server.py fedavg.py secure_agg.py curious_server.py run_summary.py dashboard.html \
       config.json setup_caddy.sh start_server.sh stop_server.sh posefl/
   cd posefl
   ```
4. **Set up HTTPS** (installs Caddy and gets the certificate; it prints the server address when it works):
   ```bash
   bash setup_caddy.sh
   ```

### Each time you train

1. Open the **SSH** window, then start the server with a password of your choice:
   ```bash
   cd posefl
   bash start_server.sh YOUR_PASSWORD --round-timeout 0
   ```
   `--round-timeout 0` makes the first phone wait for the other as long as needed (good when training apart). Other
   options work as usual: `--secure`, `--sweep sensor_study`, `--sweep secure_study`, `--round-timeout 10` for the
   dropout test. Running it again restarts the server with the new options; `bash stop_server.sh` stops it. It keeps
   running after you close the SSH window. `tail -f server.log` shows its output (Ctrl-C to stop watching).
2. **Phones:** Server URL `https://34-1-2-3.sslip.io` (the address the script printed, starting with `https://`),
   **Server password**, then **Save and test connection**. Both phones press **Start** on *Federated training*.
3. **Dashboard:** open the same `https://...` address in a browser and enter the password when asked.

### When the server code changes

The machine keeps the files you uploaded. After a new version of the code, upload the server files again with
**Upload file** (they arrive in your home folder), move them over the old ones and restart. For version 0.7 that is
these six (three of them are new):
```bash
mv fl_server.py fedavg.py secure_agg.py curious_server.py run_summary.py dashboard.html ~/posefl/ 2>/dev/null; cd ~/posefl
bash start_server.sh YOUR_PASSWORD --round-timeout 0
```
Earlier runs in `~/posefl/runs/` are kept and appear in the dashboard's *Compare runs* table.
The phones and the server must be the same version: install the new APK on both phones too.

### Getting the results back for the analysis

```bash
cd ~/posefl && tar czf runs.tgz runs
```
In the SSH window click **Download file**, enter `posefl/runs.tgz`. On the Mac, in `Sources/server`:
`tar xzf ~/Downloads/runs.tgz`, then `python3 analyze_run.py --latest N` as usual.

### When something does not work

| Problem | Fix |
|---|---|
| Phone: "the server redirects to https://..." | The URL starts with `http://`; use the `https://` address |
| Phone: "password is wrong or missing" | Enter the same password the server was started with |
| Phone: "Could not reach the server" | Check the server is running (`tail server.log`) and the address has no typo |
| `setup_caddy.sh` says the certificate is not ready | Tick *Allow HTTP traffic* and *Allow HTTPS traffic* on the VM (Edit), wait a minute, run it again. Log: `sudo journalctl -u caddy --no-pager \| tail -30` |
| The external IP changed (after stopping and starting the VM) | Run `bash setup_caddy.sh` again and use the new address on the phones |

When the project is graded, delete the VM so nothing keeps running on your card.
