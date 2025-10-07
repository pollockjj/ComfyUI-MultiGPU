#!/usr/bin/env python3
"""Convert MultiGPU JSON log into a Markdown summary."""

import argparse
import json
from pathlib import Path
from typing import Iterator, Dict, Any


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


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize MultiGPU JSON logs into Markdown")
    parser.add_argument("logfile", type=Path, help="Path to JSONL log produced by MGPU_JSON_LOG_PATH")
    parser.add_argument("--severity", nargs="*", help="Optional severity levels to include (e.g. INFO WARN ERROR)")
    parser.add_argument(
        "--component",
        nargs="*",
        help="Optional component names to include (matches component or event_category fields)",
    )
    args = parser.parse_args()

    entries = list(load_json_lines(args.logfile))
    if not entries:
        print("No entries found in log file.")
        return 0

    print("| Timestamp | Level | Component | Message |")
    print("| --- | --- | --- | --- |")
    for entry in entries:
        level = entry.get("level", "")
        if args.severity and level not in args.severity:
            continue
        component_values = {
            entry.get("component", ""),
            entry.get("event_category", ""),
        }
        component = next((value for value in component_values if value), "")
        if args.component and component not in args.component:
            continue
        timestamp = entry.get("timestamp", "")
        message = entry.get("message", "").replace("|", "\u2502")
        print(f"| {timestamp} | {level} | {component} | {message} |")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
