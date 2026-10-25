# Wi-Fi WIDS / WIPS prototype: deauthentication flood detection with a Random Forest

Reads 802.11 frame logs, turns them into per-second, per-device features, classifies each
(second, device) as normal or malicious with a scikit-learn Random Forest, raises alerts, and
(WIPS mode) automatically blocklists the offending MAC.

**Use only on your own network or an approved, isolated lab testbed.**

## Quick start

```bash
pip install -r requirements.txt

python simulate.py --out data/frames_train.csv.gz --runs 120 --duration 45 --seed 1   # synthetic training data
python train.py                                                                         # accuracy, confusion matrix, model
python simulate.py --out data/frames_demo.csv --runs 1 --duration 60 --attack-fraction 1.0 --seed 777

python detect.py --replay data/frames_demo.csv --speed 20                  # WIDS: detect + alert
python detect.py --replay data/frames_demo.csv --speed 20 --wips \
                 --protected 62:18:99:74:f6:52                              # WIPS: also blocklist
```
`--speed 20` replays at 20x real time so the demo looks live. Use `--speed 1` for real time.

## How it works (the 60-second explanation)

| File | Job |
|---|---|
| `simulate.py` | Writes a synthetic frame log (CSV rows only, no radio, nothing transmitted). Includes busy clients, scanners, legitimate one-off deauths, AP-MAC spoofing and low-rate attacks. |
| `capture.py` | Reads a **real** pcap (or live frames) into the same schema; can label a pcap from time ranges for training. |
| `features.py` | Groups frames into 1-second windows per transmitter MAC and computes 8 features. |
| `train.py` | Trains Random Forest + Decision Tree, evaluates on **held-out capture runs**, saves model and plots. |
| `detect.py` | Replays a log or sniffs live, scores each window, prints/logs alerts, and in `--wips` mode blocklists the MAC. |

Features: frames/s, deauth+disassoc frames/s, fraction of deauth frames, fraction broadcast, fraction
management, number of distinct receivers, mean RSSI, RSSI standard deviation.

Label rule: a (second, MAC) window is "malicious" if it contains at least 3 attack frames.

Why the split is by run: if windows from the same capture land in both train and test, the model
sees near-copies of its test data and scores look better than they are.

## Using real data instead of the simulator

* **Your own lab capture:** put an adapter in monitor mode on a fixed channel, capture with
  `tcpdump`/Wireshark, run your authorised test, then
  `python capture.py lab.pcapng data/lab.csv.gz --attack 30 60` (seconds during which the deauth
  frames were sent) and `python train.py --data data/lab.csv.gz`.
* **AWID3:** convert a subset to the frame-log schema in `features.py` (time, src, dst, type,
  subtype, rssi, label) and pass it to `train.py`.
* **Live:** `sudo python detect.py --iface wlan0mon --wips --protected <your AP MAC>`.
  Live and pcap paths in `capture.py` were not exercised against real radio traffic here.

## Limits worth stating honestly

* Scores on synthetic data are optimistic. Real captures are messier. Retrain on real data.
* A blocklist of attacker MACs does not stop injected deauth frames (the attacker is not associated
  to your AP, and MACs are trivially random or spoofed). The real fix is 802.11w Protected
  Management Frames. The blocklist file is the "simple automatic action"; in a lab it can feed
  hostapd's `deny_mac_file`.
* If the flagged MAC is your own AP, the attack is spoofing it. The WIPS therefore never blocks
  `--protected` MACs; it flags them instead.
* Overall accuracy is misleading because ~96% of windows are normal; judge by precision, recall and the
  confusion matrix.
