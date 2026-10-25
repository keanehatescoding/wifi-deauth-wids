"""Turn a log of 802.11 frames into one row of simple features per (second, transmitter).

Frame log schema (one row per frame):
    time     float   seconds (any origin; only differences matter)
    src      str     transmitter MAC (addr2)
    dst      str     receiver MAC (addr1)
    type     int     802.11 frame type    (0 = management, 1 = control, 2 = data)
    subtype  int     802.11 frame subtype (management: 8 beacon, 4 probe req, 10 disassoc, 12 deauth)
    rssi     float   signal strength in dBm (NaN if the radiotap header had none)
    run      int     (optional) capture/run id, used to keep train/test splits honest
    label    int     (optional) 1 if this individual frame is an attack frame (ground truth)
"""
import numpy as np
import pandas as pd

BCAST = "ff:ff:ff:ff:ff:ff"
DEAUTH_SUBTYPES = (10, 12)       # disassociation + deauthentication
MIN_ATTACK_FRAMES = 3            # a (second, MAC) window is "malicious" if it holds >= 3 attack frames

FEATURES = [
    "fps",           # frames per second sent by this MAC
    "deauth_rate",   # deauth/disassoc frames per second
    "frac_deauth",   # share of this MAC's frames that are deauth/disassoc
    "frac_bcast",    # share sent to the broadcast address
    "frac_mgmt",     # share that are management frames
    "n_uniq_dst",    # number of different receivers
    "rssi_mean",     # mean signal strength (dBm)
    "rssi_std",      # spread of signal strength (a spoofed MAC has two "locations")
]


def build_windows(df: pd.DataFrame, window: float = 1.0) -> pd.DataFrame:
    """Aggregate frames into per-(run, window, src) feature rows."""
    d = df.copy()
    if "run" not in d:
        d["run"] = 0
    d["win"] = np.floor(d["time"] / window).astype(np.int64)
    d["is_deauth"] = ((d["type"] == 0) & d["subtype"].isin(DEAUTH_SUBTYPES)).astype(int)
    d["is_bcast"] = (d["dst"] == BCAST).astype(int)
    d["is_mgmt"] = (d["type"] == 0).astype(int)

    agg = dict(
        n_frames=("time", "size"),
        n_deauth=("is_deauth", "sum"),
        n_bcast=("is_bcast", "sum"),
        n_mgmt=("is_mgmt", "sum"),
        n_uniq_dst=("dst", "nunique"),
        rssi_mean=("rssi", "mean"),
        rssi_std=("rssi", "std"),
    )
    if "label" in d:
        agg["n_attack"] = ("label", "sum")

    w = d.groupby(["run", "win", "src"], sort=False).agg(**agg).reset_index()
    w["fps"] = w["n_frames"] / window
    w["deauth_rate"] = w["n_deauth"] / window
    w["frac_deauth"] = w["n_deauth"] / w["n_frames"]
    w["frac_bcast"] = w["n_bcast"] / w["n_frames"]
    w["frac_mgmt"] = w["n_mgmt"] / w["n_frames"]
    w["rssi_mean"] = w["rssi_mean"].fillna(-100.0)
    w["rssi_std"] = w["rssi_std"].fillna(0.0)
    w["t_start"] = w["win"] * window
    if "n_attack" in w:
        w["y"] = (w["n_attack"] >= MIN_ATTACK_FRAMES).astype(int)
    return w
