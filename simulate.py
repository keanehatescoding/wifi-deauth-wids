"""Generate a SYNTHETIC frame log (no radio needed) with normal Wi-Fi traffic and deauth floods.

This writes rows into a CSV; it does not create or transmit any 802.11 frames. It exists so the
ML pipeline can be developed and demonstrated without a radio. It is a stand-in for real data
(AWID3, or your own monitor-mode captures of an isolated lab network you are authorised to test).

Realistic difficulties are built in, so the classifier has something to learn:
  * busy clients (hundreds of frames/s) and probe-request scanners (not attacks)
  * legitimate one-off deauth/disassoc frames (AP reboot, roaming)
  * attackers that spoof the AP's MAC (so the source MAC alone proves nothing)
  * low-rate attacks (3-8 frames/s) that overlap with legitimate behaviour
"""
import argparse
import numpy as np
import pandas as pd

BCAST = "ff:ff:ff:ff:ff:ff"


def rand_mac(rng):
    b = rng.integers(0, 256, 6)
    b[0] = (b[0] & 0xFE) | 0x02  # unicast, locally administered
    return ":".join(f"{x:02x}" for x in b)


def poisson_times(rng, rate, t0, t1):
    n = rng.poisson(rate * (t1 - t0))
    return np.sort(rng.uniform(t0, t1, n))


def frames(times, src, dst, ftype, subtype, rssi_mu, rssi_sd, label, rng):
    n = len(times)
    dst_arr = np.full(n, dst, dtype=object) if isinstance(dst, str) else rng.choice(dst, n)
    sub = np.full(n, subtype) if np.isscalar(subtype) else rng.choice(subtype[0], n, p=subtype[1])
    return pd.DataFrame({
        "time": times,
        "src": np.full(n, src, dtype=object),
        "dst": dst_arr,
        "type": np.full(n, ftype),
        "subtype": sub,
        "rssi": rng.normal(rssi_mu, rssi_sd, n).round(0),
        "label": np.full(n, label),
    })


def simulate_run(rng, run_id, duration, with_attack):
    parts = []
    aps = [rand_mac(rng) for _ in range(rng.integers(1, 3))]
    ap_rssi = {ap: rng.uniform(-55, -35) for ap in aps}

    # Beacons: ~10 per second from every AP
    for ap in aps:
        t = np.arange(0, duration, 0.1) + rng.normal(0, 0.003, int(duration / 0.1))
        parts.append(frames(np.sort(t), ap, BCAST, 0, 8, ap_rssi[ap], 1.5, 0, rng))

    # Clients with very different traffic levels
    clients = []
    for _ in range(rng.integers(3, 7)):
        c, ap = rand_mac(rng), aps[rng.integers(len(aps))]
        clients.append((c, ap))
        c_rssi = rng.uniform(-78, -38)
        rate = rng.choice([5, 20, 50, 120, 300], p=[.35, .30, .20, .10, .05])
        parts.append(frames(poisson_times(rng, rate, 0, duration), c, ap, 2, 0, c_rssi, 3, 0, rng))
        parts.append(frames(poisson_times(rng, rate * .8, 0, duration), ap, c, 2, 0, ap_rssi[ap], 1.5, 0, rng))
        parts.append(frames(poisson_times(rng, .2, 0, duration), c, BCAST, 0, 4, c_rssi, 3, 0, rng))
        # legitimate disconnects: an AP sends 1-4 deauth/disassoc frames to one client
        if rng.random() < .35:
            t0 = rng.uniform(5, duration - 5)
            n = rng.integers(1, 5)
            parts.append(frames(np.sort(rng.uniform(t0, t0 + .5, n)), ap, c, 0,
                                ([12, 10], [.7, .3]), ap_rssi[ap], 1.5, 0, rng))

    # Hard negative: probe-request scanner (noisy but not an attack)
    if rng.random() < .35:
        t0 = rng.uniform(0, duration - 30)
        parts.append(frames(poisson_times(rng, rng.uniform(30, 100), t0, t0 + rng.uniform(10, 30)),
                            rand_mac(rng), BCAST, 0, 4, -70, 3, 0, rng))
    # Hard negative: AP restart sends a couple of broadcast deauths
    if rng.random() < .25:
        ap = aps[0]
        t0 = rng.uniform(5, duration - 5)
        parts.append(frames(np.sort(rng.uniform(t0, t0 + .3, rng.integers(1, 4))), ap, BCAST, 0, 12,
                            ap_rssi[ap], 1.5, 0, rng))

    # Deauthentication flood(s)
    if with_attack:
        for _ in range(rng.integers(1, 3)):
            kind = rng.choice(["broadcast", "targeted", "low_rate"], p=[.35, .40, .25])
            t0 = rng.uniform(8, duration - 30)
            t1 = t0 + rng.uniform(8, 22)
            rate = {"broadcast": rng.uniform(15, 300),
                    "targeted": rng.uniform(10, 250),
                    "low_rate": rng.uniform(3, 8)}[kind]
            spoof = rng.random() < .6
            src = aps[rng.integers(len(aps))] if spoof else rand_mac(rng)
            dst = BCAST if kind == "broadcast" else clients[rng.integers(len(clients))][0]
            parts.append(frames(poisson_times(rng, rate, t0, t1), src, dst, 0,
                                ([12, 10], [.8, .2]), rng.uniform(-80, -40), 2.5, 1, rng))

    df = pd.concat(parts, ignore_index=True).sort_values("time", kind="stable").reset_index(drop=True)
    df.insert(0, "run", run_id)
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output CSV (use .csv.gz to compress)")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--attack-fraction", type=float, default=0.5, help="share of runs containing an attack")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    runs = [simulate_run(rng, i, a.duration, rng.random() < a.attack_fraction) for i in range(a.runs)]
    df = pd.concat(runs, ignore_index=True)
    df.to_csv(a.out, index=False)
    print(f"wrote {len(df):,} frames from {a.runs} runs -> {a.out}  "
          f"({int(df['label'].sum()):,} attack frames)")


if __name__ == "__main__":
    main()
