# Step by step plan (your part, mapped to the outline and the proposal)

Today is Nov 13. Proposal timeline says Nov 4-17 is "FedAvg and local vs global comparison", Nov 18-25 is
sensor study plus extra credit, Nov 26-Dec 1 is new-user tests and demo rehearsal, Dec 1-3 is report and
submission. Outline deadlines: report and Sources due Dec 3. Demo: the outline says Dec 5 in one place and
Dec 1/3 in the table. Ask the TA which one is real.

Grade weights to keep in mind: demo 80% (ranked by live accuracy on other people), report 15%, code 5%.
Extra credit cannot replace required parts. So accuracy of the live system comes first.

## Step 0: sync with Hasin (today, 15 min)
- [ ] Send him docs/PROTOCOL.md. Agree it is the contract (features, seeds, API, file layout).
- [ ] Decide who owns the server. The proposal gives Hasin the server and you the check against a reference.
      `python/fedpose/server` is a complete working server. Either he adopts it, or he keeps his and you run
      `check_kotlin_parity.py` against it.
- [ ] Decide who owns int8 encoding (his task). It already exists in `Wire.kt` and `wire.py`. He reviews it.

## Step 1: get the toolchain working (laptop)
- [ ] `cd python && pip install -r requirements.txt && python scripts/make_synth_data.py --out ../data_synth && python -m pytest tests -q`
- [ ] Start the server, open `http://localhost:8080/status` in a browser. (README section 2)

## Step 2: first run on the phones (outline checklist: two Android clients and a server)
- [ ] Open `android/` in Android Studio, run the app on both phones. Fix any Gradle error (paste it to me).
- [ ] Both phones on the laptop's Wi-Fi, enter `http://<laptop-ip>:8080`.
- [ ] Phone with you = Client B (your assignment). Set "This phone is" to Client B.

## Step 3: collect Client B data (your assignment, outline section 3)
Record in the app. Train split only for poses 3,4,5. Vary tilt, surface, hand-held vs resting, slight movement.
- [ ] train: portrait_down, landscape_left, landscape_right, at least 4 sessions each, mix of users and holds
- [ ] val: all six poses, 1-2 sessions each (used only to pick settings)
- [ ] test: all six poses, 2+ sessions each (never used for tuning)
- [ ] newuser: all six poses recorded by a person who recorded NO training data (outline: unseen users test your demo)
- [ ] Use "Show recorded data summary" to check counts. Pull the folder with `adb pull` (README section 3).

## Step 4: run FedAvg and the three-model comparison (outline sections 5-7)
- [ ] Both phones: Step 1 (local-only baseline), Step 2 (join federated training), Step 3 (evaluate).
- [ ] Server writes `server_data/rounds.jsonl` and `reports.jsonl`.
- [ ] `python scripts/make_plots.py --results ../results --out ../figures --reports server_data/reports.jsonl`
      gives phone-reported confusion matrices (`phone_eval.md`).
- [ ] Expect: each local model tops out near 50% accuracy on six poses (cannot predict poses it never saw), global is much higher.
      That is the answer to "what does federated learning provide that local training alone cannot".

## Step 5: offline experiments for the report (3 seeds)
- [ ] Put both phones' pulled data in `data/raw/A/...` and `data/raw/B/...`.
- [ ] `python scripts/run_experiments.py --data ../data --out ../results all`
- [ ] Tuning uses validation data only. Read `results/chosen_config.json`, use the same settings on the phones (epochs, lr, rounds).
- [ ] Parity check: `python scripts/check_kotlin_parity.py ...` proves the phone code and the Python experiments compute the same thing.

## Step 6: required technical contribution = sensor study (outline section 8)
Already implemented as `sensor` in run_experiments.py: acc | acc+grav | +gyro | +rotation vector, everything else fixed.
- [ ] Look at `figures/fig_sensor_study.png` and the table. Report the winner on the UNSEEN-USER column.
- [ ] Expected talking point: rotation-vector values depend on compass heading (yaw), so they can hurt generalization.
      Check whether your data says that. Do not assume it.
- [ ] To use another feature set live: start the server with `--features 6|12|18|24` and choose the same feature set in both apps.

## Step 7: extra credit (up to 20 points, text says 20 in one place and 30 in another)
Implemented: int8 compression, adaptive participation, dropout recovery. Simulated comparison in `extra_credit.json`,
real-system tests with `check_dropout_live.py` and the app's "Adaptive participation" box.
- [ ] Live dropout demo: during federated training turn Wi-Fi off on one phone for a few rounds, turn it back on.
      Phone log shows retries or "rejected (stale)", then it catches up. Server log shows `reason: timeout`.
- [ ] Honest findings to state: compression saves little in absolute terms because the model has 78 numbers and
      HTTP/JSON overhead dominates. Adaptive participation saves bytes, not computation (the update must be computed to
      know it is small). During an outage the global model drifts toward the one client still training.

## Step 8: demo prep (80% of grade)
- [ ] Live prediction: Step 4 saves the global model on each phone. Use "Start live prediction" with model "global".
- [ ] Rehearse with people who gave no training data. Fix mistakes by recording more varied data and rerunning rounds.
- [ ] Rehearse: record, local train, federate, three-model comparison on the same pose, extension, findings. 15-30 min incl. Q&A.
- [ ] Both team members present. Bring a USB cable and a hotspot as Wi-Fi backup. Test the laptop firewall in the actual room.

- [ ] Read `docs/DEMO_DAY.md` end to end. It maps every rubric line to something on screen and has the Q&A answers.
- [ ] In the app, section 5: "Load all three models" on the phone that will be tested, then "Open blind-test screen". Do 36 practice placements with a friend.

## Step 9: report (5-10 pages) and Sources folder
Your sections per the task division: on-device training and evaluation. Learning outcomes section must list what each of you did.
- [ ] Sources/ = this repo minus data and results. README already has the steps. Fill in exact versions.
- [ ] Figures in `figures/`, tables in `figures/results_tables.md`.
- [ ] Say in the report that offline experiments used CSVs pulled over USB from your own phones while the deployed system keeps raw data on the phones.
