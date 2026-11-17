# Protocol contract (share this with Hasin)

Everything the phones, the server and the Python experiments must agree on. If you change any
of it, change `python/fedpose/constants.py` and `android/core/.../Constants.kt` together, then
rerun `python scripts/check_kotlin_parity.py`. It fails loudly if the two sides drift apart.

## Poses and clients
Class index = model output index: 0 screen_up, 1 screen_down, 2 portrait_up, 3 portrait_down,
4 landscape_left, 5 landscape_right. Client A trains on 0,1,2. Client B trains on 3,4,5. Both
record val, test and newuser data for all six poses.

## Raw session file (one CSV per recording, stays on the phone)
Path: `raw/<A|B>/<train|val|test|newuser>/<pose>__<user>__<nn>.csv`

    t_ns,sensor,x,y,z
    5123456789,acc,0.01,0.02,9.80

`t_ns` = Android `SensorEvent.timestamp`. `sensor` in `acc` (TYPE_ACCELEROMETER), `grav`
(TYPE_GRAVITY), `gyro` (TYPE_GYROSCOPE), `rot` (TYPE_ROTATION_VECTOR, x y z only).
Sessions are split by FILE, then windowed, so overlapping windows never cross a split.

## Windows and features
1 s windows, 0.5 s stride, first and last 1 s of a session dropped, a sensor needs at least 10
samples in a window. Per sensor: mean(x,y,z) then std(x,y,z) (population std) of values divided by a
FIXED scale: acc and grav by 9.81, gyro by pi, rot by 1. No per-client standardization (it would
differ between clients under non-IID data and hurt FedAvg).

| feature set | sensors | features | params |
|---|---|---|---|
| acc | acc | 6 | 42 |
| acc_grav (baseline) | acc, grav | 12 | 78 |
| acc_grav_gyro | + gyro | 18 | 114 |
| acc_grav_gyro_rot | + rot | 24 | 150 |

## Model
Softmax regression. Parameter vector = W row-major (class-major, 6 x nf) then b (6).
Minibatch SGD, batch 32, cross entropy, optional L2 and FedProx term. Shuffling uses SplitMix64
(same code in both languages) so a seed gives identical results in Kotlin and Python.
Seeds: round r, client c: `base*1000003 + r*1009 + idx(c)` (A=0, B=1). Local baseline:
`base*1000003 + 777 + idx(c)`, trained for `rounds * local_epochs` epochs from zeros.

## Server API (JSON over HTTP, same Wi-Fi)
| call | purpose |
|---|---|
| `POST /register {client_id}` | returns n_features, max_rounds, status. Phone refuses to start if its feature count differs |
| `GET /model?client_id=A` | status (idle/training/done), round, version, weights (base64 float32 LE), must_participate, already_submitted, round_deadline_s |
| `POST /update` | one upload per client per round (below). 200 accepted, 409 stale, 400 malformed |
| `GET /round_status` | who has answered this round |
| `POST /local_model {client_id, weights}` and `GET /local_model?client_id=A&asker=B` | server relays the two local-only models so each phone can score all three (weights only) |
| `POST /report {client_id, model, test_set, accuracy, macro_f1, confusion, ...}` | metrics from a phone's own evaluation. Never data |
| `GET /status`, `GET /results`, `POST /admin/start {max_rounds, reset, n_features}` | monitoring and manual start. `n_features` (6, 12, 18, 24) switches the feature set live for the contribution demo |
| `GET /audit`, `GET /audit.html` | privacy evidence: every upload the server ever received, with its key names, byte size and class counts. The HTML page auto refreshes, put it on the projector |

### Update body
    {"client_id":"A","round":3,"base_version":2,"n_samples":504,"skipped":false,
     "codec":"f32"|"int8","scale":0.0123,"payload":"<base64>","train_loss":0.4,"train_acc":0.9,
     "class_counts":[170,170,164,0,0,0]}
* `payload` is the DELTA (local weights minus the global weights the phone received).
* f32: float32 little endian, 4 B per parameter. int8: `scale = max|d|/127` (stored as float32),
  `q = rint(d/scale)` clipped to +-127, 1 B per parameter.
* Skip notice (adaptive participation): `{"client_id","round","base_version","n_samples":0,"skipped":true,"update_norm":0.07}`.

### Server rules
* Round closes when both expected clients answered, or after `--timeout` seconds (dropout).
* Aggregation: `new = global + sum(n_i * delta_i) / sum(n_i)` over received updates (= sample weighted FedAvg).
  Nobody contributed (all skipped or timed out): model kept, round still advances.
* Version check: `round` and `base_version` must equal the server's current values, otherwise 409 with
  the current ones. The phone re-syncs with `GET /model` and retrains on the latest model.
* A repeated upload for the same round is acknowledged but counted once (safe retries).
* Scheduling: after `--max-skips` consecutive skips the server answers `must_participate: true`.
* Traffic: server counts HTTP body bytes (and an estimate of header bytes) per client and per round
  (`server_data/rounds.jsonl`, `GET /status`). Phones count their own body bytes too.

## Privacy guard
The server accepts only whitelisted keys. `/update`: client_id, round, base_version, n_samples, skipped, codec, scale,
payload, train_loss, train_acc, class_counts, update_norm. `/report`: client_id, model, test_set, accuracy, macro_f1, n,
per_class_recall, per_class_f1, confusion. Any other key is dropped and noted in the audit log, so a raw sensor array
cannot sneak in. `class_counts` is how many windows of each pose a phone trained on (counts only, no sensor values).

## Blind test (demo day)
A placement is a 2 second clip. The phone cuts 1 s windows at start offsets 0.5 s and 1.0 s (falls back to 0, 0.25, 0.5
if the clip is short), computes the same features as training, averages the softmax probabilities of the windows and takes
the argmax. This is done for the global model and both local models. `python/fedpose/blind.py` and
`android/core/.../BlindTest.kt` give identical probabilities (checked to 1e-16).
