# Demo day run sheet (20 min slot) mapped to the professor's rubric

Slot plan: 7 min blind test, 8 min presentation, 5 min Q&A. Extra credit goes in the 3 to 5 min buffer.
Roles: Shiv runs the phones and the blind test. Hasin runs the laptop (server, /audit.html, slides). Swap if you prefer.

## T minus 30 min checklist
- [ ] Laptop on the same Wi-Fi as both phones. Firewall allows port 8080. Hotspot ready as backup.
- [ ] Server running: `python -m fedpose.server --host 0.0.0.0 --port 8080 --rounds 30 --timeout 60 --autostart`
- [ ] Browser tab 1: `http://<ip>:8080/audit.html` (privacy evidence). Tab 2: `/status`.
- [ ] Both phones: record fresh data on site if the rubric wants it, train local, run federated rounds until done.
- [ ] Phone being tested: section 5 "Load all three models" shows global, local A and local B loaded.
- [ ] Phone screen brightness up, auto lock off, volume off, battery above 50%.
- [ ] Practice the blind test 36 times with someone who gave no training data (use `python scripts/blind_test_sim.py --split --trials 36` for a sanity estimate; try `--trim 0` for the pessimistic case).

## Blind test (7 min)
Rubric Part A: 30 x global accuracy + 5 x local A accuracy on A poses + 5 x local B accuracy on B poses.
1. Judges pick the pose cards and hold the phone. Open "Open blind-test screen".
2. For each of the 36 placements: judge presses START, 3 s countdown, 2 s recording, screen shows global, local A, local B predictions and the true pose they entered.
3. Keep the phone still for the whole 2 s. Do not touch the screen during recording.
4. At the end the screen shows the tally and the rubric score. `blind_log.csv` is saved on the phone.
Expected: global high, each local model around 50 percent overall (right on its own three poses, 0 percent on the other three). That is the point, say it out loud.

## Privacy evidence (rubric item)
- Show `/audit.html`: each row is one upload with keys, bytes and class counts. No sensor arrays, only a weight delta.
- Say: "The server only accepts a whitelist of keys. Raw samples have nowhere to go."
- On the phone, section 5 "Show last upload" prints exactly what that phone sent last.
- Class counts show non-IID: A uploaded counts only for poses 0 to 2, B only for 3 to 5.

## Presentation (8 min, suggested split)
1. 1 min: problem and why federated (sensor data is private, 2 phones, 6 poses, non-IID split).
2. 1 min: architecture (phone trains softmax on 12 features, uploads delta, server FedAvg weighted by samples).
3. 2 min: live federated rounds on both phones, server log and /audit.html side by side.
4. 2 min: three-model comparison table (global vs local A vs local B) and confusion matrices.
5. 2 min: contribution live (below).

## Contribution live (sensor study)
Show that features matter: from the laptop, run
`curl -X POST localhost:8080/admin/start -H 'Content-Type: application/json' -d '{"n_features":6,"reset":true}'`
then the same with 12, 18, 24 (each phone must have its feature spinner set to match, the phone refuses on mismatch).
Message: acc only (6) is worse, adding gravity (12) is the jump, gyro and rotation vector (18, 24) add nothing for static poses.
Backup: show the figure from `figures/` if time runs out.

## Scripted Q&A answers (say them conversationally)
**Why does federated beat local?**
Each phone only saw three of the six poses, so a local model has never been told the other three exist. Its logits for those classes only ever get pushed down, so it gets them wrong every time and tops out near 50 percent. FedAvg averages the two updates so the global model learns all six poses without either phone sharing data.

**Why are mean gravity features enough?**
A static pose is defined by which device axis gravity points along: plus or minus x, y or z. The six poses sit in different spots of that 3D gravity vector, so they are close to linearly separable and a softmax (linear) model handles them. The mean over a 1 second window removes hand jitter and the std tells it how steady the phone is.

**Why is the gyroscope useless here?**
The gyroscope measures rotation rate, not orientation. In a static pose it reads about zero for every class, so it carries no class information. It only adds noise features. Our sensor study shows no gain from adding it.

**Why does the local model get unseen poses wrong, not just uncertain?**
Softmax training never sees those labels, so their weights and biases are only decreased. The model confidently picks one of its own three classes.

**What does the server see?**
Weight deltas, sample counts, class counts and metrics. Never sensor values. The audit page proves it.

**What if one phone drops out?**
Round timeout, then the server aggregates what it has. A stale upload is rejected with 409 and the phone resyncs. The global model drifts toward the remaining client and recovers when the other returns (we tested this live).

**Does int8 compression help?**
Honestly very little in absolute bytes. The model is only 78 numbers, so HTTP and JSON overhead dominates. It would matter for bigger models.

## If something breaks
- Phone cannot reach server: check same Wi-Fi, firewall, or switch to the hotspot and retype the IP.
- Feature mismatch on register: set the spinner on both phones to the server's `n_features`.
- Blind-test screen says model missing: run "Load all three models" again, or retrain local on that phone.
- Predictions all one class: phone orientation calibration issue, re-record a few varied sessions before the demo, not during it.
