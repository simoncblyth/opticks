#!/usr/bin/env python

import os, textwrap


def metrics_truncate():
    """
    # Opening and closing in 'w' mode truncates the file
    """
    METRICS_PATH = os.environ.get("METRICS_PATH", None)
    if METRICS_PATH is None: return
    with open(METRICS_PATH, "w") as fp:
        pass
    pass

def metrics_append_gauge(gauge="opticks_sim_duration_seconds", value=0, desc="Simulation time"):
    """
    """
    METRICS_PATH = os.environ.get("METRICS_PATH", None)
    if METRICS_PATH is None: return

    OJ_TEST_NAME = os.environ.get("OJ_TEST_NAME", "no_OJ_TEST_NAME")

    labels = f'test="{OJ_TEST_NAME}",host="{os.uname()[1]}"'

    txt = textwrap.dedent(f"""\
    # HELP {gauge} {desc}
    # TYPE {gauge} gauge
    {gauge}{{{labels}}} {value}

    """)

    try:
        with open(METRICS_PATH, "a") as fp: fp.write(txt)
    except IOError as e:
        print(f"Warning: Could not write metrics to {METRICS_PATH}: {e}")
    pass



