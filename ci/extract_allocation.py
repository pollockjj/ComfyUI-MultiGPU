#!/usr/bin/env python3
"""Filter MultiGPU JSON logs for allocation summaries."""

import argparse
import json
from pathlib import Path
from typing import Iterable, Iterator, Dict, Any


def load_json_lines(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def is_allocation_event(entry: Dict[str, Any], keywords: Iterable[str]) -> bool:
    message = entry.get("message", "")
    return any(keyword in message for keyword in keywords)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract allocation-related events from MultiGPU JSON logs")
    parser.add_argument("logfile", type=Path, help="Path to JSONL log produced by MGPU_JSON_LOG_PATH")
    parser.add_argument(
        "--keywords",
        nargs="*",
        default=["Final Allocation String", "Total memory", "Virtual VRAM"],
        help="Keywords that mark allocation events",
    )
    args = parser.parse_args()

    entries = list(load_json_lines(args.logfile))
    if not entries:
        print("No entries found in log file.")
        return 0

    matched = [entry for entry in entries if is_allocation_event(entry, args.keywords)]
    if not matched:
        print("No allocation events matched provided keywords.")
        return 0

    for entry in matched:
        timestamp = entry.get("timestamp", "unknown")
        category = entry.get("event_category", "")
        component = entry.get("component", "")
        header_bits = [bit for bit in (timestamp, category, component) if bit]
        header = " | ".join(header_bits) if header_bits else "allocation"
        print(f"## {header}")
        print(entry.get("message", ""))
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
