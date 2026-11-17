# Federated phone-pose recognition (CSC 8223)

Two Android phones learn six phone poses together without sharing sensor data. Raw sensor files
and extracted windows stay on each phone. The server only receives model deltas, sample counts,
and evaluation metrics.

```
android/core   pure Kotlin: features, softmax SGD, int8 codec, HTTP client, live predictor, JVM test harness
android/app    Android app: record sessions, train local, federate, evaluate, live predict
python/fedpose reference implementation, FL server, offline simulator, metrics, synthetic data
python/scripts experiments, plots, parity and dropout checks
docs/          PROTOCOL.md (the contract), STEPS.md (what to do, in order)
```

## Versions used while developing
Python 3.13 (3.10+ works), numpy 2.5, Flask 3.1, matplotlib 3.11, pytest 9. Kotlin 2.4 compiler and JDK 21
for the JVM tests. Android: Android Studio with AGP 8.5.2, Kotlin 2.0.20, Gradle 8.7, JDK 17, compileSdk 34,
minSdk 26. Update this list with the exact versions on your machines before you submit.

## 1. Python setup and tests
```bash
cd python
pip install -r requirements.txt
python scripts/make_synth_data.py --out ../data_synth      # SYNTHETIC data, for testing only
python -m pytest tests -q                                  # 15 tests, includes server vs reference simulator
```

## 2. Run the server
```bash
cd python
python -m fedpose.server --host 0.0.0.0 --port 8080 --rounds 30 --timeout 60 --autostart
# --features 12 is the default (acc_grav). Use 6, 18 or 24 for the other feature sets.
```
Find the laptop's Wi-Fi IP (`ipconfig` / `ifconfig`) and type `http://<ip>:8080` into both phones.
Windows firewall must allow inbound port 8080 on the private network.
Training starts when both A and B have joined (`--autostart`) or with `curl -X POST localhost:8080/admin/start`.
Privacy evidence page for the demo: `http://<ip>:8080/audit.html`. Live feature switch for the contribution:
`curl -X POST localhost:8080/admin/start -H 'Content-Type: application/json' -d '{"n_features":18,"reset":true}'`.
Demo day script and Q&A answers: `docs/DEMO_DAY.md`. Rehearse the blind test offline: `python scripts/blind_test_sim.py --split --trials 36`.

## 3. Build and run the Android app
Open the `android/` folder in Android Studio, let Gradle sync, run `app` on each phone (USB debugging on).
Not compiled in the authoring environment: if Gradle complains, send me the exact error.
In the app: choose A or B, record sessions, then Step 1, Step 2, Step 3, Live.
Recorded data folder is shown in the log. Pull it for offline experiments:
```bash
adb pull /sdcard/Android/data/com.fedpose.app/files ./phone_A_data     # run once per phone (use -s <serial>)
```
Merge into one tree `data/raw/A/...` and `data/raw/B/...` (the folders already contain `raw/<client>/...`).

## 4. Kotlin tests without a phone (JVM)
```bash
cd android && ./gradlew :core:run --args="selftest"
# or with a plain kotlinc:  kotlinc core/src/main/kotlin -include-runtime -d /tmp/fedpose-core.jar
#                           java -cp /tmp/fedpose-core.jar com.fedpose.cli.VirtualPhoneKt selftest
cd ../python
python scripts/check_kotlin_parity.py --jar /tmp/fedpose-core.jar --data ../data_synth
python scripts/check_dropout_live.py  --jar /tmp/fedpose-core.jar --data ../data_synth
```

## 5. Experiments and figures
```bash
cd python
python scripts/run_experiments.py --data ../data --out ../results all
python scripts/make_plots.py --results ../results --out ../figures --reports server_data/reports.jsonl
```
`results/` JSON and `figures/` (PNG + `results_tables.md`) feed the report directly.
Offline experiments use the CSVs pulled from the team's own phones. The deployed system never uploads them.
