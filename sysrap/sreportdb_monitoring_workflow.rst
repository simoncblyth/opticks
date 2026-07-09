sreportdb_monitoring_workflow
================================

Practical Usage
----------------

Gitlab-CI orchestrates scheduled test running with FAILs notified by email.
Tests needing GPU are handled by gitlab-runners polling into gitlab from GPU workstations.
The first three stages listed in the below sections are all orchestrated by Gitlab-CI,
the fourth presentation stage is handled by the Grafana web application running on the GPU workstation.
Grafana is a widely used open source interactive data visualization web application that
provides flexible interactive editing of "dashboard" visualizations.


* https://code.ihep.ac.cn/blyth/oj/-/pipeline_schedules
* https://code.ihep.ac.cn/blyth/oj/-/pipelines
* http://localhost:3000/d/ad25xw2/cxs-timings?orgId=1&from=now-2d&to=now&timezone=browser&var-ci_pipeline_id=33391&var-ci_job_id=124041&var-TargetRun=5

  * gitlab-ci logging provides grafana url links for each test


Current DB Schema : versionset/runs/events (most fields elided for brevity)
---------------------------------------------------------------------------
::

      +---------------------------+
      |    opticks_versionset     |   (Software Stack / Hardware Baseline)
      +---------------------------+
      | [PK] id                   |
      |      UNIQUE(versions...)  |
      +---------------------------+
                   |  1
                   |
                   |  M
      +---------------------------+
      |       opticks_runs        |   (GitLab-CI or Local Benchmark Run)
      +---------------------------+
      | [PK] id                   |
      | [FK] versionset_id        |--------> REFERENCES opticks_versionset(id)
      |      ci_pipeline_id       |
      |      ci_job_id            |   ===> UNIQUE INDEX (pipeline, job) [if CI]
      |      run_timestamp        |   ===> UNIQUE INDEX (timestamp)    [if Local]
      +---------------------------+
                   |  1
                   |
                   |  M (ON DELETE CASCADE)
      +---------------------------+
      |      opticks_events       |   (Individual Ray-Traced Events)
      +---------------------------+
      | [PK] id                   |
      | [FK] run_id               |--------> REFERENCES opticks_runs(id)
      |      event_index          |
      +---------------------------+


As more tests types are added to the monitoring the schema will need additions.

TODO: Investigate using json blob(s) within the schema which together with
      sqlite3 json_extract could avoid frequent changes at SQL level.



First Stage : running - writes potentially large SEvt and metadata
--------------------------------------------------------------------

Opticks scripts invoke executables that optionally write:

1. events (SEvt.hh) into folders(NPFold.h) of (NP.hh) arrays in NumPy format

   * full details of every step of every photon can be saved - which can be many gigabytes - far too much data to keep longterm

2. SProf.txt time and CPU memory profiling with count annotations

   * profiling files are small, making them suited for longterm persisting


Second Stage : reporting - summarizes SEvt and metadata into persisted sreport (run and evsmry arrays)
--------------------------------------------------------------------------------------------------------

Some tests such as cxs_min.sh accept the "report" argument which runs the sreport binary.
For details see sources::

   ~/o/sysrap/sreport.h
   ~/o/sysrap/sreport_Creator.h
   ~/o/sysrap/tests/sreport.cc
    ~/np/NP.hh

Crucial groups of methods::

    sreport_Creator::init_runprof_run_ranges_from_SProf
    NP::MakeMetaKVS_ranges2

    sreport_Creator::init_evsmry_from_ranges
    NP::Summarize_ranges_to_evsmry

The sreport_Creator uses NPFold::LoadNoData to load folder and event metadata
creating the sreport instance. The sreport binary is typically invoked
from the LOGDIR of opticks runs and the report(which is small) is persisted
into ${LOGDIR}_sreport directory.

If SREPORT_ARCHIVE_DIR is defined sreport also invokes sreport::save_into_archive


Third Stage : reportdb - ingest summary info extracted from reports (run and evsmry arrays) into SQLite3 database tables
--------------------------------------------------------------------------------------------------------------------------

For details see::

    ~/o/sysrap/tests/sreportdb.sh
    ~/o/sysrap/tests/sreportdb.cc   ## simple main creating sreportdb instance from
    ~/o/sysrap/sreportdb.sql        ## table schema
    ~/o/sysrap/sreportdb.h          ## extracts info from run and evsmry arrays and metadata to populate sqlite3 database using C API
    ~/np/NSQLite.h


Currently the database is recreated and fully repopulated from all the report dir on every run.
This makes it straightforward to change schema while the workflow is expected to need frequent
changes from incorporation of new tests.  After most tests have been adapted and the workflow
has stabilized a more efficient approach of just adding new reports to the database can be used.


Q: Could the 2nd and 3rd stages be combined ?
A: NO, keeping reportdb stage entirely separate from report is useful to isolate DB translation
   from reporting.  The stages can be viewed as progressive summarizations. The later stages aim
   to provide an overview to monitor for validation failures or performance changes. Earlier
   stages give detail needed for debugging.

   The "reportdb" stage should be kept as a simple translation from array data and metadata into tables.
   For investigating issues the report stage and full SEvt stage are essential and need to be kept
   useful and unrelated to the DB stage.

   For reportdb and presentation stages the point is to summarize aggressively,
   leaving just metrics that can signal an issue, without the detail needed to solve the problem.



Fourth Stage : presentation - Grafana dashboard to show Opticks running metrics
-----------------------------------------------------------------------------------

Grafana with sqlite3 datasource plugin provides graphical dashboard presenting metrics collected into the
sreportdb database by the above workflow.


How to integate a test into this workflow ?
--------------------------------------------

Clone OJ repository which houses config and scripts::

    git clone git@code.ihep.ac.cn:blyth/oj.git

NB the OJ scripts act as wrappers around scripts and executables
from Opticks and JUNOSW installations. The wrappers aim to
standardize script running and outputs to facilitate Gitlab-CI usage.


1. add script and config to OJ repo, following the pattern of other tests, see::

   * ~/oj/.gitlab-ci.yml

2. ensure that info needed for presentation is propagated along the running-reporting-reportdb workflow,
   which will likely requires changes to sreport and sreportdb headers and database schema from the
   Opticks repository

3. add Grafana dashboard panels for the new test which present metrics that
   show validity and performance of the test


Alternatively : start prototype additions using python
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Could use python sqlite module to create a separate sqlite3 database,
which can be combined with the existing Grafana dashboard via
adding a separate data source.

BUT to facilitate integration should try to reuse the existing schema
as much as possible::

   ~/o/sysrap/sreportdb.h





