"""Read real 802.11 traffic (pcap file or live monitor-mode interface) into the frame-log schema.

Use ONLY on your own network or an approved, isolated lab testbed.

Convert a pcap to a labelled CSV for training (label deauth/disassoc frames inside the time
ranges, in seconds from the first frame, where you ran your authorised test):
    python capture.py lab.pcapng data/lab.csv.gz --attack 30 60 --attack 100 120
"""
import argparse

import pandas as pd

BCAST = "ff:ff:ff:ff:ff:ff"


def frame_row(pkt):
    """One scapy packet -> dict in the frame-log schema (None if it is not a usable 802.11 frame)."""
    from scapy.layers.dot11 import Dot11, RadioTap
    if not pkt.haslayer(Dot11):
        return None
    d = pkt[Dot11]
    if d.addr2 is None:          # control frames such as ACK have no transmitter address
        return None
    rssi = getattr(pkt[RadioTap], "dBm_AntSignal", None) if pkt.haslayer(RadioTap) else None
    return dict(time=float(pkt.time), src=d.addr2.lower(), dst=(d.addr1 or BCAST).lower(),
                type=int(d.type), subtype=int(d.subtype),
                rssi=float(rssi) if rssi is not None else float("nan"))


def read_pcap(path):
    from scapy.utils import PcapReader
    rows = []
    with PcapReader(path) as reader:
        for pkt in reader:
            r = frame_row(pkt)
            if r:
                rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["time"] -= df["time"].min()
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pcap")
    ap.add_argument("out")
    ap.add_argument("--attack", nargs=2, type=float, action="append", metavar=("START", "END"),
                    help="seconds (from first frame) during which deauth/disassoc frames are labelled attack")
    a = ap.parse_args()
    df = read_pcap(a.pcap)
    df["run"] = 0
    df["label"] = 0
    for t0, t1 in a.attack or []:
        m = df["time"].between(t0, t1) & (df["type"] == 0) & df["subtype"].isin([10, 12])
        df.loc[m, "label"] = 1
    df.to_csv(a.out, index=False)
    print(f"{len(df):,} frames, {int(df['label'].sum()):,} labelled attack -> {a.out}")


if __name__ == "__main__":
    main()
