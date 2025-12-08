#!/usr/bin/env python3
"""
SProf.py
========
Enhanced parser for JUNO/Opticks timing logs with:
- microseconds since epoch
- VM and RSS in KB → displayed in MB
- optional comments after " # "
- time delta from first stamp
- time delta from previous stamp

Now with additional summary table for all __HEAD / __TAIL instrumented methods
- Automatically detects nested/sub-methods
- Strips A000_/B001_ etc. event prefixes for cleaner names
- Shows POST→DOWN sub-duration when available
- Picks the first comment found within the method (usually on BRES or TAIL)
"""

import sys
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

LEVEL = int(os.environ.get("LEVEL", "0"))

@dataclass(order=True)
class Stamp:
    """Single timing stamp with all parsed and computed data"""
    ts_micro: int
    vm_kb: int
    rss_kb: int
    label: str
    dt: datetime

    comment: str = ""
    delta_s: float = 0.0          # seconds since first stamp
    delta_prev: float = 0.0       # seconds since previous stamp


def format_memory(kb: int) -> str:
    return f"{kb / 1024:.2f}"


def get_timezone() -> timezone:
    """Robust TZ parsing: supports UTC, +8, +08, -5, 8, etc."""
    TZ = os.environ.get("TZ", "+08").strip().upper()

    if TZ == "UTC":
        return timezone.utc

    # Handle optional leading sign
    sign = 1
    if TZ.startswith(('-', '+')):
        sign = -1 if TZ.startswith('-') else 1
        TZ = TZ[1:]
    else:
        sign = 1

    # Pad single digit like "8" → 8, "08" → 8
    hours = int(TZ)
    return timezone(sign * timedelta(hours=hours))


def parse_timing_file(filepath: str) -> list[Stamp]:
    """Parse the file and return a list of Stamp objects (deltas not yet computed)"""
    tz = get_timezone()
    stamps: list[Stamp] = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith('#') or ':' not in line:
                continue

            try:
                label_part, rest = line.split(':', 1)
                label = label_part.strip()

                comment = ""
                if ' # ' in rest:
                    rest, comment_part = rest.split(' # ', 1)
                    comment = " # " + comment_part.rstrip()

                parts = [x.strip() for x in rest.split(',', 3)]
                if len(parts) < 3:
                    continue

                ts_micro = int(parts[0])
                vm_kb = int(parts[1])
                rss_kb = int(parts[2])

                dt = datetime.fromtimestamp(ts_micro / 1e6, tz=tz)

                stamps.append(Stamp(ts_micro=ts_micro,
                                   vm_kb=vm_kb,
                                   rss_kb=rss_kb,
                                   label=label,
                                   dt=dt,
                                   comment=comment))
            except ValueError:
                print(f"Warning: Skipping malformed line: {raw_line.rstrip()}", file=sys.stderr)

    return stamps


def compute_deltas(stamps: list[Stamp]) -> None:
    """In-place computation of delta_s and delta_prev"""
    if not stamps:
        return

    first_ts = stamps[0].ts_micro
    prev_ts = stamps[0].ts_micro

    stamps[0].delta_s = 0.0
    stamps[0].delta_prev = 0.0

    for stamp in stamps[1:]:
        stamp.delta_s = (stamp.ts_micro - first_ts) / 1e6
        stamp.delta_prev = (stamp.ts_micro - prev_ts) / 1e6
        prev_ts = stamp.ts_micro


def print_report(stamps: list[Stamp], tz_name: str) -> None:
    """Print the nice table + summary"""
    if not stamps:
        print("No valid timing stamps found in file.")
        return

    header = f"DateTime ({tz_name})"
    print(f"{'Delta(s)':>10} {'Δprev(s)':>10} {header:19} {'VM(MB)':>8} {'RSS(MB)':>8} Label / Comment")
    print("-" * 92)

    for s in stamps:
        dt_str = s.dt.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        print(f"{s.delta_s:10.6f} {s.delta_prev:10.6f} {dt_str} "
              f"{format_memory(s.vm_kb):>8} {format_memory(s.rss_kb):>8} {s.label}{s.comment}")


def print_summary(stamps: list[Stamp], tz_name: str) -> None:
    # ───── Summary of instrumented methods (HEAD → TAIL) ─────
    marker_dict = defaultdict(dict)
    for s in stamps:
        if '__' in s.label:
            base, marker = s.label.rsplit('__', 1)
            if LEVEL > 0: print("base[%s] marker[%s]" % (base, marker))
            marker_dict[base][marker] = s
        pass
    pass

    #print("\nSummary of instrumented methods (HEAD → TAIL)")
    print(f"{'Method':50} {'Start(s)':>10} {'Duration(s)':>12} {'Sub PREL→POST(s)':>20} {'Sub POST→DOWN(s)':>20} {'VM(MB)':>8} {'RSS(MB)':>8} Comment")
    print("-" * 150)

    summaries = []
    for base in marker_dict:
        if LEVEL > 0: print("marker_dict.base[%s]" % base)
        marks = marker_dict[base]
        if LEVEL > 0: print("marks.keys:%r" % repr(marks.keys()))

        headkey, tailkey = 'simulate_HEAD', 'reset_HEAD'
        if not (headkey in marks and tailkey in marks):
            continue

        head = marks[headkey]
        tail = marks[tailkey]
        if LEVEL > 0: print("marker_dict.head[%s].tail[%s]" % (repr(head), repr(tail)))

        duration = tail.delta_s - head.delta_s

        post_down = 'simulate_POST', 'simulate_DOWN'
        post_down_dur_str = ""
        if post_down[0] in marks and post_down[1] in marks:
            post_down_dur = marks[post_down[1]].delta_s - marks[post_down[0]].delta_s
            post_down_dur_str = f"{post_down_dur:16.6f} (POST→DOWN)"
        pass

        prel_post = 'simulate_PREL', 'simulate_POST'
        prel_post_dur_str = ""
        if prel_post[0] in marks and prel_post[1] in marks:
            prel_post_dur = marks[prel_post[1]].delta_s - marks[prel_post[0]].delta_s
            prel_post_dur_str = f"{prel_post_dur:16.6f} (PREL→POST)"
        pass


        # Pick first comment found inside the method (BRES, TAIL, etc.)
        comment = next((s.comment for s in marks.values() if s.comment), "")

        #nice_name = re.sub(r'^[A-Z0-9]+_', '', base).replace('__', '::')
        nice_name = "%s__%s->%s" % ( base, headkey,tailkey)

        summaries.append((head.delta_s, nice_name, duration, prel_post_dur_str, post_down_dur_str,
                         format_memory(tail.vm_kb), format_memory(tail.rss_kb), comment))

    summaries.sort(key=lambda x: x[0])  # sort by start time

    for start_s, name, dur, prel_post_str, post_down_str, vm, rss, com in summaries:
        print(f"{name:50} {start_s:10.3f} {dur:12.6f} {prel_post_str:>20} {post_down_str:>20} {vm:>8} {rss:>8} {com}")


def parse_tz():
    # Determine timezone display string
    raw_tz = os.environ.get("TZ", "+08").strip()
    if raw_tz.upper() == "UTC":
        tz_display = "UTC"
    else:
        # Normalise to +08 style
        sign = ""
        if raw_tz.startswith(('-', '+')):
            sign = raw_tz[0]
            raw_tz = raw_tz[1:]
        else:
            sign = "+"
        try:
            tz_display = f"{sign}{int(raw_tz):02d}"
        except:
            tz_display = raw_tz or "+08"
        pass
    pass
    return tz_display


if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "SProf.txt"

    stamps = parse_timing_file(filepath)
    compute_deltas(stamps)

    tz_display = parse_tz()

    #print_report(stamps, tz_display)
    print_summary(stamps, tz_display)
pass
