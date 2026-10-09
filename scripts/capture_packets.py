"""Capture live packets to PCAP for downstream flow extraction."""
from __future__ import annotations
import argparse
import time
from pathlib import Path

def main() -> None:
    try:
        from scapy.all import AsyncSniffer, wrpcap
    except ImportError as exc:
        raise SystemExit("Install optional capture support: pip install -r requirements-capture.txt") from exc
    parser = argparse.ArgumentParser(description="Capture live network traffic to PCAP")
    parser.add_argument("--interface", default=None)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--output", default="data/live.pcap")
    args = parser.parse_args()
    if args.seconds <= 0:
        raise SystemExit("--seconds must be > 0")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sniffer = AsyncSniffer(iface=args.interface, store=True)
    print(f"Capturing on {args.interface or 'default interface'} for {args.seconds:.1f}s...")
    sniffer.start()
    time.sleep(args.seconds)
    packets = sniffer.stop()
    wrpcap(str(output), packets)
    print(f"Captured {len(packets)} packets -> {output}")
    print("Convert the PCAP to the trained flow-feature schema before inference.")

if __name__ == "__main__":
    main()
