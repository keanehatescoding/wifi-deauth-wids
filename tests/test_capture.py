import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from capture import read_pcap


def test_read_pcap_empty_returns_schema(monkeypatch):
    class FakeReader:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            return iter([])

    monkeypatch.setattr("scapy.utils.PcapReader", lambda path: FakeReader())
    df = read_pcap("fake.pcap")

    assert len(df) == 0
    assert list(df.columns) == ["time", "src", "dst", "type", "subtype", "rssi"]
