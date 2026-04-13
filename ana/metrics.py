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


def junit_write_report(duration, success, error="", log_snippet=""):
    """

    * https://github.com/testmoapp/junitxml


    Results from .gitlab-ci.yml job-name "opticks:external-test-job:gcc11:el9" appear at::

         /blyth/oj/-/pipelines/29129/test_report?job_name=opticks%3Aexternal-test-job%3Agcc11%3Ael9

    A table is presented with columns::

        Suite
        Name
        Filename
        Status
        Duration
        Details  : useless repetition of : Suite,Name,Duration

    junit info that is not presented::

        system-out
        system-err


    """
    JUNIT_PATH = os.environ.get("JUNIT_PATH", None)
    if JUNIT_PATH is None: return

    OJ_TEST_NAME = os.environ.get("OJ_TEST_NAME", "no_OJ_TEST_NAME")
    safe_log = log_snippet.replace("]]>", "]]&gt;")

    testsuite = "opticks"
    classname = "does-classname-appear-in-interface"
    relpath = os.environ.get("TESTSCRIPT", "oj3/oj3.sh")
    status_tag = "" if success else f'<failure message="FAIL">{error}</failure>'
    tests = 1
    failures = 0 if success else 1


    content = textwrap.dedent(f"""\
        <?xml version="1.0" encoding="UTF-8"?>
        <testsuites>
            <testsuite name="{testsuite}" tests="{tests}" failures="{failures}" >
                <testcase name="{OJ_TEST_NAME}" classname="{classname}" file="{relpath}" time="{duration}">
                    {status_tag}
                    <system-out><![CDATA[{safe_log}]]></system-out>
                    <system-err><![CDATA[{safe_log}]]></system-err>
                </testcase>
            </testsuite>
        </testsuites>
    """).strip()

    try:
        with open(JUNIT_PATH, "w") as fp: fp.write(content)
    except IOError as e:
        print(f"Warning: Could not write to {JUNIT_PATH}: {e}")
    pass





