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



Example SProf.txt file::

    A[blyth@localhost detsim]$ cat /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec03_ok1_hit_seed42_evtmax1_run/SProf.txt
    SEvt__Init_RUN_META:1764832256883411,1047572,507772
    CSGOptiX__Create_HEAD:1764832286732555,6547320,1409368
    CSGOptiX__Create_TAIL:1764832287998570,8713612,2011600
    junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD:1764832362774663,11365756,4661272
    A000_QSim__simulate_HEAD:1764832362774729,11365756,4661272
    A000_SEvt__BeginOfRun:1764832362774764,11365756,4661272
    A000_SEvt__beginOfEvent_FIRST_EGPU:1764832362774931,11365756,4661272
    A000_SEvt__setIndex:1764832362774957,11365756,4661272
    A000_QSim__simulate_LBEG:1764832362873445,11420500,4715772
    A000_QSim__simulate_PRUP:1764832362873480,11420500,4715772
    A000_QSim__simulate_PREL:1764832362934980,29475668,4716668
    A000_QSim__simulate_POST:1764832385992249,29475668,4721148
    A000_QSim__simulate_DOWN:1764832387907191,31192664,6439040
    A000_QSim__simulate_LEND:1764832387907234,31192664,6439040
    A000_QSim__simulate_PCAT:1764832387907257,31192664,6439040
    A000_QSim__simulate_BRES:1764832387907301,31192664,6439040 # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    A000_QSim__simulate_TAIL:1764832387907307,31192664,6439040
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_HEAD:1764832387907480,31192664,6439040
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_HEAD:1764832387907545,31192664,6439040
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_TAIL:1764832418223584,38489020,13688040
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_TAIL:1764832418223644,38489020,13688040
    A000_junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_TAIL:1764832418223651,38489020,13688040
    A000_QSim__reset_HEAD:1764832418224055,38489020,13688040
    A000_SEvt__endIndex:1764832418224348,38434276,13633476
    A000_SEvt__EndOfRun:1764832418228430,36717280,11916480
    A[blyth@localhost detsim]$


::

    A[blyth@localhost detsim]$ REPORT=1 ~/o/bin/SProf.py /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec03_ok1_hit_seed42_evtmax1_run/SProf.txt
      Delta(s)   Δprev(s) DateTime (+08)        VM(MB)  RSS(MB) Label / Comment
    --------------------------------------------------------------------------------------------
      0.000000   0.000000 2025-12-04 15:10:56.883  1023.02   495.87 SEvt__Init_RUN_META
     29.849144  29.849144 2025-12-04 15:11:26.732  6393.87  1376.34 CSGOptiX__Create_HEAD
     31.115159   1.266015 2025-12-04 15:11:27.998  8509.39  1964.45 CSGOptiX__Create_TAIL
    105.891252  74.776093 2025-12-04 15:12:42.774 11099.37  4552.02 junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_HEAD

    105.891318   0.000066 2025-12-04 15:12:42.774 11099.37  4552.02 A000_QSim__simulate_HEAD
    105.891353   0.000035 2025-12-04 15:12:42.774 11099.37  4552.02 A000_SEvt__BeginOfRun
    105.891520   0.000167 2025-12-04 15:12:42.774 11099.37  4552.02 A000_SEvt__beginOfEvent_FIRST_EGPU
    105.891546   0.000026 2025-12-04 15:12:42.774 11099.37  4552.02 A000_SEvt__setIndex
    105.990034   0.098488 2025-12-04 15:12:42.873 11152.83  4605.25 A000_QSim__simulate_LBEG
    105.990069   0.000035 2025-12-04 15:12:42.873 11152.83  4605.25 A000_QSim__simulate_PRUP
    106.051569   0.061500 2025-12-04 15:12:42.934 28784.83  4606.12 A000_QSim__simulate_PREL
    129.108838  23.057269 2025-12-04 15:13:05.992 28784.83  4610.50 A000_QSim__simulate_POST
    131.023780   1.914942 2025-12-04 15:13:07.907 30461.59  6288.12 A000_QSim__simulate_DOWN
    131.023823   0.000043 2025-12-04 15:13:07.907 30461.59  6288.12 A000_QSim__simulate_LEND
    131.023846   0.000023 2025-12-04 15:13:07.907 30461.59  6288.12 A000_QSim__simulate_PCAT
    131.023890   0.000044 2025-12-04 15:13:07.907 30461.59  6288.12 A000_QSim__simulate_BRES # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928
    131.023896   0.000006 2025-12-04 15:13:07.907 30461.59  6288.12 A000_QSim__simulate_TAIL
    131.024069   0.000173 2025-12-04 15:13:07.907 30461.59  6288.12 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_HEAD
    131.024134   0.000065 2025-12-04 15:13:07.907 30461.59  6288.12 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_HEAD
    161.340173  30.316039 2025-12-04 15:13:38.223 37586.93 13367.23 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_cpumerged_TAIL
    161.340233   0.000060 2025-12-04 15:13:38.223 37586.93 13367.23 A000_junoSD_PMT_v2_Opticks__EndOfEvent_CollectFullHits_TAIL
    161.340240   0.000007 2025-12-04 15:13:38.223 37586.93 13367.23 A000_junoSD_PMT_v2_Opticks__EndOfEvent_Simulate_TAIL
    161.340644   0.000404 2025-12-04 15:13:38.224 37586.93 13367.23 A000_QSim__reset_HEAD

    161.340937   0.000293 2025-12-04 15:13:38.224 37533.47 13313.94 A000_SEvt__endIndex
    161.345019   0.004082 2025-12-04 15:13:38.228 35856.72 11637.19 A000_SEvt__EndOfRun
    Method                                               Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)   VM(MB)  RSS(MB) Comment
    ------------------------------------------------------------------------------------------------------------------------------------------------------
    A000_QSim__simulate_HEAD->reset_HEAD                  105.891    55.449326        23.057269 (PREL→POST)         1.914942 (POST→DOWN) 37586.93 13367.23  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928

    echo 161.340644 - 105.891318 | bc -l  # 55.449326



A000_QSim__simulate_HEAD => A000_QSim__reset_HEAD
    head and tail of a single events simulation including collection



This reproduces the "hit" line numbers::

    A[blyth@localhost OJ_LOCAL_Dec04_ok1_hit_seed42_evtmax1_run]$ ~/o/bin/SProf.py
    Method                                               Start(s)  Duration(s)     Sub PREL→POST(s)     Sub POST→DOWN(s)    Sub TAIL→RESET(s)    VM(MB)  RSS(MB)  Comment
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
    A000_QSim__simulate_HEAD->reset_HEAD                  106.466   215.560469            22.996241             1.949322           190.446287   32168.83  7971.96  # numGenstepCollected=583922,numPhotonCollected=148793197,numHit=27471928

    A[blyth@localhost OJ_LOCAL_Dec04_ok1_hit_seed42_evtmax1_run]$ pwd
    /data1/blyth/tmp/j/zhenning_double_muon/detsim/OJ_LOCAL_Dec04_ok1_hit_seed42_evtmax1_run




"""

import sys
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path

LEVEL = int(os.environ.get("LEVEL", "0"))

@dataclass(order=True)
class Stamp:
    """
    Single timing stamp with all parsed and computed data

    label
       identifier for location in code where SProf::Stamp was used

    ts_micro
       microseconds since the epoch
    vm_kb
       virtual memory
    rss_kb
       resident set size

    comment
       optional context string

    """
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
    """
    Parse the file and return a list of Stamp objects (deltas not yet computed)
    """
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
    """
    In-place computation of delta_s and delta_prev
    """
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


def print_report(stamps: list[Stamp], tz_name: str, idx: int) -> None:
    """
    Print the nice table + summary
    """
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


def print_summary(stamps: list[Stamp], tz_name: str, idx: int, fid: str) -> None:
    # ───── Summary of instrumented methods (HEAD → TAIL) ─────
    """
    Prints summary of instrumented methods giving:

    * HEAD -> TAIL
    * PREL -> POST
    * POST -> DOWN


    1. Create two level marker_dict where the base keys are struct names
       such as "QSim" and the inner keys are location-in-method-labels
       such as "simulate_HEAD"

    2. Iterate over base keys looking for ones with location keys
       "simulate_HEAD" and "reset_HEAD" giving "head" and "tail" marks

    3. When head and tail are found look for "simulate_POST" "simulate_DOWN"

    4. When head and tail are found also look for "simulate_PREL" "simulate_POST"


    """
    marker_dict = defaultdict(dict)
    for s in stamps:
        if '__' in s.label:
            base, marker = s.label.rsplit('__', 1)
            if LEVEL > 0: print("base[%s] marker[%s]" % (base, marker))
            marker_dict[base][marker] = s
        pass
    pass

    if idx == 0:
        print(f"{'Identity':50} {'Start(s)':>10} {'Duration(s)':>12} {'Sub PREL→POST(s)':>20} {'Sub POST→DOWN(s)':>20} {'Sub TAIL→RESET(s)':>20}  {'VM(MB)':>8} {'RSS(MB)':>8} {'Comment':>8}")
        wid = 50 + 1 + 10 + 1 + 12 + 1 + 20 + 1 + 20 + 1 + 20 + 2 + 8 + 1 + 8 + 1 + 8
        print("-" * wid )
    pass


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
            post_down_dur_str = f"{post_down_dur:16.6f}" #" (POST→DOWN)"
        pass

        prel_post = 'simulate_PREL', 'simulate_POST'
        prel_post_dur_str = ""
        if prel_post[0] in marks and prel_post[1] in marks:
            prel_post_dur = marks[prel_post[1]].delta_s - marks[prel_post[0]].delta_s
            prel_post_dur_str = f"{prel_post_dur:16.6f}" #" (PREL→POST)"
        pass


        tail_reset = 'simulate_TAIL', 'reset_HEAD'
        tail_reset_dur_str = ""
        if tail_reset[0] in marks and tail_reset[1] in marks:
            tail_reset_dur = marks[tail_reset[1]].delta_s - marks[tail_reset[0]].delta_s
            tail_reset_dur_str = f"{tail_reset_dur:16.6f}" # " (TAIL→RESET)"
        pass


        # Pick first comment found inside the method (BRES, TAIL, etc.)
        #comment = next((s.comment for s in marks.values() if s.comment), "")

        comments = []
        for s in marks.values():
            if s.comment: comments.append(s.comment)
        pass
        comment = " ".join(reversed(comments))

        #nice_name = re.sub(r'^[A-Z0-9]+_', '', base).replace('__', '::')
        #nice_name = "%s__%s->%s" % ( base, headkey,tailkey)
        nice_name = "%s/%s" % ( fid, base)


        smry = (head.delta_s,
                nice_name,
                duration,
                prel_post_dur_str,
                post_down_dur_str,
                tail_reset_dur_str,
                format_memory(tail.vm_kb),
                format_memory(tail.rss_kb),
                comment)
        summaries.append(smry)
    pass

    summaries.sort(key=lambda x: x[0])  # sort by start time
    for start_s, name, dur, prel_post_str, post_down_str, tail_reset_str, vm, rss, com in summaries:
        print(f"{name:50} {start_s:10.3f} {dur:12.6f} {prel_post_str:>20} {post_down_str:>20} {tail_reset_str:>20}   {vm:>8} {rss:>8} {com}")
    pass


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
    rawpath = sys.argv[1] if len(sys.argv) > 1 else "SProf.txt"

    fp = Path(rawpath).resolve()
    fold = fp.parent.name
    fid = fold
    for fe in fold.split("_"):
        if fe.startswith("hit"): fid = fe
    pass
    #print(fold)

    idx = int(os.environ.get("SPROF_INDEX", 0))

    stamps = parse_timing_file(fp)
    compute_deltas(stamps)

    tz_display = parse_tz()

    if "REPORT" in os.environ:
        print_report(stamps, tz_display, idx)
    pass

    print_summary(stamps, tz_display, idx, fid)
pass
