"""WIDS / WIPS: score traffic window-by-window with the trained model and raise alerts.

Replay a frame log (demo, works anywhere):
    python detect.py --replay data/frames_demo.csv --speed 20
    python detect.py --replay data/frames_demo.csv --speed 20 --wips --protected <your AP MAC>

Live, on a monitor-mode interface of YOUR OWN / approved lab network (needs root + scapy):
    sudo python detect.py --iface wlan0mon --wips --protected aa:bb:cc:dd:ee:ff

IDS mode (default) prints and logs alerts. WIPS mode (--wips) also adds the offending MAC to a
blocklist file (one MAC per line, which is also the hostapd `deny_mac_file` format), unless the MAC
is in --protected (your own APs). A protected MAC sending a deauth flood means someone is SPOOFING
it, and blocking it would block your own AP, so that case is only flagged.
"""
import argparse
import json
import math
import threading
import time
from collections import defaultdict

import joblib
import pandas as pd

from features import FEATURES, build_windows


class Responder:
    def __init__(self, alert_log, blocklist, protected, wips, cooldown, label_fn):
        self.alert_log, self.blocklist = alert_log, blocklist
        self.protected, self.wips, self.cooldown = {m.lower() for m in protected}, wips, cooldown
        self.label = label_fn
        self.last_alert, self.suppressed, self.blocked = {}, defaultdict(int), set()
        self.n_alerts = 0

    def handle(self, r, prob):
        src, ts = r.src, r.t_start
        if src in self.last_alert and ts - self.last_alert[src] < self.cooldown:
            self.suppressed[src] += 1        # same source still flooding: don't spam
            return
        self.last_alert[src] = ts
        self.n_alerts += 1

        action = "alert only (IDS mode)"
        if self.wips:
            if src in self.protected:
                action = "NOT blocked: protected MAC, probably spoofed - check signal strength / investigate"
            elif src in self.blocked:
                action = "already blocklisted"
            else:
                self.blocked.add(src)
                with open(self.blocklist, "a") as f:
                    f.write(src + "\n")
                action = f"MAC added to {self.blocklist}"

        print(f"[ALERT] {self.label(ts)}  deauth flood suspected  src={src}  "
              f"deauth/s={r.deauth_rate:.0f}  fps={r.fps:.0f}  bcast={r.frac_bcast:.0%}  "
              f"rssi={r.rssi_mean:.0f}dBm  confidence={prob:.2f}\n          action: {action}", flush=True)
        with open(self.alert_log, "a") as f:
            f.write(json.dumps(dict(time=self.label(ts), src=src, deauth_per_s=r.deauth_rate, fps=r.fps,
                                    rssi_mean=r.rssi_mean, rssi_std=r.rssi_std, confidence=round(float(prob), 3),
                                    action=action)) + "\n")


def score_window(frames, model, resp, window, threshold):
    w = build_windows(frames, window)
    if w.empty:
        return 0
    w["p"] = model.predict_proba(w[FEATURES])[:, 1]
    for r in w[w["p"] >= threshold].sort_values("t_start").itertuples():
        resp.handle(r, r.p)
    return len(w)


def replay(path, model, resp, window, threshold, speed):
    df = pd.read_csv(path)
    if "run" not in df:
        df["run"] = 0
    df["_w"] = (df["time"] // window).astype(int)
    n_frames = len(df)
    for _, chunk in df.groupby(["run", "_w"], sort=True):
        score_window(chunk.drop(columns="_w"), model, resp, window, threshold)
        if speed > 0:
            time.sleep(window / speed)
    return n_frames


def live(iface, model, resp, window, threshold):
    from scapy.sendrecv import AsyncSniffer
    from capture import frame_row
    buf, lock = [], threading.Lock()

    def cb(pkt):
        r = frame_row(pkt)
        if r:
            with lock:
                buf.append(r)

    sniffer = AsyncSniffer(iface=iface, prn=cb, store=False)
    sniffer.start()
    print(f"sniffing on {iface} (Ctrl-C to stop)", flush=True)
    try:
        while True:
            time.sleep(window)
            cutoff = math.floor(time.time() / window) * window   # only score fully elapsed windows
            with lock:
                done = [r for r in buf if r["time"] < cutoff]
                buf[:] = [r for r in buf if r["time"] >= cutoff]
            if done:
                score_window(pd.DataFrame(done), model, resp, window, threshold)
    except KeyboardInterrupt:
        pass
    finally:
        sniffer.stop()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--replay", help="frame-log CSV to replay")
    src.add_argument("--iface", help="monitor-mode interface for live capture")
    ap.add_argument("--model", default="results/wids_rf.joblib")
    ap.add_argument("--window", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=0.5, help="alert when P(attack) >= this")
    ap.add_argument("--speed", type=float, default=0, help="replay speed multiplier (0 = as fast as possible)")
    ap.add_argument("--wips", action="store_true", help="enable prevention: blocklist offending MACs")
    ap.add_argument("--protected", nargs="*", default=[], help="your own AP MACs (never blocklisted)")
    ap.add_argument("--cooldown", type=float, default=10, help="seconds between repeat alerts for one MAC")
    ap.add_argument("--alert-log", default="results/alerts.jsonl")
    ap.add_argument("--blocklist", default="results/blocked_macs.txt")
    a = ap.parse_args()

    model = joblib.load(a.model)
    label_fn = (lambda t: f"t={t:6.0f}s") if a.replay else (lambda t: time.strftime("%H:%M:%S", time.localtime(t)))
    resp = Responder(a.alert_log, a.blocklist, a.protected, a.wips, a.cooldown, label_fn)

    mode = "WIPS (detect + respond)" if a.wips else "WIDS (detect only)"
    print(f"== {mode} | window {a.window}s | threshold {a.threshold} ==", flush=True)
    if a.replay:
        n = replay(a.replay, model, resp, a.window, a.threshold, a.speed)
        print(f"\nreplayed {n:,} frames -> {resp.n_alerts} alert(s) raised, "
              f"{sum(resp.suppressed.values())} repeat window(s) suppressed, {len(resp.blocked)} MAC(s) blocklisted")
    else:
        live(a.iface, model, resp, a.window, a.threshold)


if __name__ == "__main__":
    main()
