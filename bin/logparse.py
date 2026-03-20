#!/usr/bin/env python3
"""
logparse.py
============

Advanced log parser – supports two timestamp formats:

Usage::

    logparse.py /path/to/log.txt


1. parses file collecting Entry objects for lines with timestamps in the below formats:

  * 2025-11-24 11:08:57.661 hello world
  * Hello World @ localhost on Mon Nov 24 14:11:24 2025


2. print the timestamped entries in time order with time differences


"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import os
import re
import sys
import textwrap

# Format 1: Standard ISO-like with milliseconds
PATTERN1 = re.compile(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3})\s+(.*)$')
FMT1 = "%Y-%m-%d %H:%M:%S.%f"

# Format 2: "Running @ host on Mon Nov 24 14:11:24 2025"
# Only lines containing " @ " are candidates
PATTERN2 = re.compile(r'(?:^|\s)(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})\s+(\d{2}:\d{2}:\d{2})\s+(\d{4})')
FMT2 = "%a %b %d %H:%M:%S %Y"  # e.g. "Mon Nov 24 14:11:24 2025"

MONTH_MAP = {
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

@dataclass(order=True)
class Entry:
    ts: datetime                  # for sorting
    ts_display: str               # how to show in output
    message: str
    source: str = ""              # optional: "ISO" or "Legacy"

    def format_ts(self) -> str:
        """Returns string like: 2025-12-04 17:12:08.508"""
        # %f gives 6 digits (microseconds), we slice to get 3 (milliseconds)
        return self.ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]




def parse_format1(line: str) -> Optional[Entry]:
    """
    :param line: of the logfile
    :return Entry/None:
    """
    m = PATTERN1.match(line.rstrip("\n"))
    if not m:
        return None
    ts_str, msg = m.groups()
    try:
        ts = datetime.strptime(ts_str, FMT1)
        return Entry(ts, ts_str, msg.strip() or "(empty)", "ISO")
    except ValueError:
        return None

def parse_format2(line: str) -> Optional[Entry]:
    """
    :param line: of the logfile
    :return Entry/None:
    """
    line_clean = line.rstrip("\n")
    if " @ " not in line_clean:
        return None

    m = PATTERN2.search(line_clean)
    if not m:
        return None

    weekday, month_str, day, time, year = m.groups()
    month = MONTH_MAP[month_str]
    day = int(day)
    full_dt_str = f"{weekday} {month_str} {day:2d} {time} {year}"
    
    try:
        ts = datetime.strptime(full_dt_str, FMT2)
        # Use original line fragment around the timestamp for display
        before = line_clean[:m.start()].rstrip()
        after = line_clean[m.end():].lstrip()
        display = f"{month_str} {day} {time} {year}"
        message = line_clean.strip()
        return Entry(ts, display, message, "Legacy")
    except ValueError:
        return None

def parse_line(line: str) -> Optional[Entry]:
    """
    Entries are yielded when the log line contains timestamps in the above handled formats

    :param line: of the logfile
    :return Entry/None:
    """
    return parse_format1(line) or parse_format2(line)

def parse_log(filepath: Path) -> List[Entry]:
    """
    :param filepath:
    :return List[Entry]: 
    """
    entries = []
    with filepath.open(encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            entry = parse_line(raw_line)
            if entry:
                entries.append(entry)
    # Sort all entries chronologically
    entries.sort(key=lambda e: e.ts)
    return entries

def print_timing_table(entries: List[Entry], source: Path):
    if not entries:
        print("No timestamped lines found.")
        return

    print(f"\nFound {len(entries)} timestamped lines in {source.name}\n")

    # Column widths
    w_ts   = 23   # longest is "2025-11-24 11:08:57.661"
    w_start = 15
    w_delta_s = 12
    w_delta_ms = 10

    header = f"{'Timestamp':<{w_ts}}  {'Since Start (s)':>{w_start}}  {'Δ Prev (s)':>{w_delta_s}}  {'Δ Prev (ms)':>{w_delta_ms}}  Message"
    print(header)
    print("-" * len(header))

    if not entries:
        return

    first_ts = entries[0].ts
    prev_ts = first_ts

    for i, e in enumerate(entries):
        since_start = (e.ts - first_ts).total_seconds()

        if i == 0:
            delta_s = 0.0
            delta_ms = 0
        else:
            delta = (e.ts - prev_ts).total_seconds()
            delta_s = delta
            delta_ms = int(round(delta * 1000))

        fixed = (
            f"{e.format_ts():<{w_ts}}  "
            f"{since_start:>{w_start}.3f}  "
            f"{delta_s:>{w_delta_s}.3f}  "
            f"{delta_ms:>{w_delta_ms}}  "
        )

        print(fixed + e.message)

        # Wrap very long messages
        if "WRAP" in os.environ and len(e.message) > 80:
            wrapper = textwrap.TextWrapper(
                initial_indent=" " * (w_ts + w_start + w_delta_s + w_delta_ms + 8),
                subsequent_indent=" " * (w_ts + w_start + w_delta_s + w_delta_ms + 8),
                width=130
            )
            for wrapped_line in wrapper.wrap(e.message)[1:]:
                print(wrapped_line)
            pass
        pass

        prev_ts = e.ts

    print()

def save_csv(entries: List[Entry], out_path: Path):
    if not entries:
        return
    first_ts = entries[0].ts
    prev_ts = first_ts
    with out_path.open("w", encoding="utf-8") as f:
        f.write("Timestamp (Display),Timestamp (ISO),Since Start (s),Δ Prev (s),Δ Prev (ms),Message,Source\n")
        for i, e in enumerate(entries):
            since_start = (e.ts - first_ts).total_seconds()
            if i == 0:
                delta_s = delta_ms = 0
            else:
                delta_s = (e.ts - prev_ts).total_seconds()
                delta_ms = int(round(delta_s * 1000))
            iso_str = e.ts.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            safe_msg = e.message.replace('"', '""')
            f.write(f'"{e.ts_display}","{iso_str}",{since_start:.6f},{delta_s:.6f},{delta_ms},"{safe_msg}","{e.source}"\n')
            prev_ts = e.ts

def main():
    if len(sys.argv) != 2:
        print("Usage: python logparse.py <logfile>")
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.is_file():
        print(f"File not found: {path}")
        sys.exit(1)

    entries = parse_log(path)
    print_timing_table(entries, path)

    csv_path = path.with_suffix(".parsed.csv")
    save_csv(entries, csv_path)
    #print(f"CSV saved → {csv_path}")

if __name__ == "__main__":
    main()
