#!/usr/bin/env python3
"""
SProf.py
========

Enhanced parser for JUNO/Opticks timing logs with:
- microseconds since epoch
- VM and RSS in KB
- optional comments after " # "
- time delta from first stamp
- time delta from previous stamp
"""

import sys
from datetime import datetime, timezone

def format_memory(kb):
    return f"{kb / 1024:.2f}"

def parse_timing_file(filepath):
    first_ts = None
    prev_ts = None

    # Header
    print(f"{'Delta(s)':>10} {'Δprev(s)':>10}  {'DateTime (UTC)':19}  {'VM(MB)':>8}  {'RSS(MB)':>8}  Label / Comment")
    print("-" * 92)

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or ':' not in line:
                continue

            # Split label and rest
            label_part, rest = line.split(':', 1)
            label = label_part.strip()

            # Handle optional comment after " # "
            comment = ""
            if ' # ' in rest:
                rest, comment = rest.split(' # ', 1)
                comment = " # " + comment

            # Extract the three numbers
            nums = [x.strip() for x in rest.split(',', 3)[:3]]
            if len(nums) < 3:
                continue

            try:
                ts_micro = int(nums[0])
                vm_kb    = int(nums[1])
                rss_kb   = int(nums[2])
            except ValueError:
                print(f"Warning: Skipping malformed line: {raw_line.rstrip()}", file=sys.stderr)
                continue

            dt = datetime.fromtimestamp(ts_micro / 1e6, tz=timezone.utc)
            dt_str = dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            # Delta from first timestamp
            if first_ts is None:
                first_ts = ts_micro
                delta_s = 0.0
            else:
                delta_s = (ts_micro - first_ts) / 1e6

            # Delta from previous timestamp
            if prev_ts is None:
                delta_prev = 0.0
            else:
                delta_prev = (ts_micro - prev_ts) / 1e6
            prev_ts = ts_micro

            # Output
            print(f"{delta_s:10.6f} {delta_prev:10.6f}  {dt_str}  "
                  f"{format_memory(vm_kb):>8}  {format_memory(rss_kb):>8}  {label}{comment}")

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "SProf.txt"
    parse_timing_file(filepath)
