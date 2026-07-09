sreportdb_grafana.rst
=======================

See also::

    ~/oj/bin/grafana.sh - permissions and datasource provisioning setup
    ~/o/sysrap/sreportdb.sql - schema

To follow how the sreportdb database is created see::

    ~/o/sysrap/sreportdb.h
    ~/o/sysrap/tests/sreportdb.cc
    ~/o/sysrap/tests/sreportdb.sh


Grafana Sharing Links
----------------------

* http://localhost:3000/d/ad25xw2/cxs-timings?orgId=1&from=now-6h&to=now&timezone=browser&var-TargetRun=5
* http://localhost:3000/public-dashboards/95767c3162484335822e588daf057a5a

Exporting dashboard as json/yaml
-----------------------------------

Right hand narrow pane, click on downarrow icon.



References for Grafana panel visualizations
--------------------------------------------

* https://grafana.com/docs/grafana/latest/visualizations/panels-visualizations/visualizations/


fr-ser grafana-sqlite-datasource
----------------------------------

* https://github.com/fr-ser/grafana-sqlite-datasource/blob/main/docs/examples.md


Grafana Web Interface shortcuts : shift+?
------------------------------------------

* hold "shift" and press "?" to bring up a panel of many key shortcuts


How to use gitlab-ci to add a non-sheduled test run - eg to check plotting results from a different test
---------------------------------------------------------------------------------------------------------

Click the right arrow icon "Run pipeline schedule" on the schedules page for an out-of-schedule run

* https://code.ihep.ac.cn/blyth/oj/-/pipeline_schedules

Inspect success of the tests on the pipelines page

* https://code.ihep.ac.cn/blyth/oj/-/pipelines

If the report stage succeeded then the sreportdb.sqlite3 will have been updated, allowing
to directly visualize results written to database from Grafana web interface

* http://localhost:3000/dashboard/new


gitlab-ci config in ~/oj/.gitlab-ci.yml
----------------------------------------

* configure what is run via :  ~/oj/.gitlab-ci.yml
* .yml variables become envvars within yml script blocks and the scripts they run

crucial variables
~~~~~~~~~~~~~~~~~~

ENVSET_NAME
   eg OK_LOCAL, OJ_LOCAL,  OJ_063 : interpreted by ~/oj/ENVSET.sh - picking which environment script to source
   selecting between releases on cvmfs or local installs on workstation of opticks(OK) or opticks+junosw(OJ)

   * NB when using local installs tests can be run with just installed versions of OK or OJ without any release
   * MUST INSTALL ANY CHANGED SCRIPTS eg by rebuilding and installing opticks with "lo;oo"

caution : multiple levels at which envvars can be set
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* gitlab-ci web interface pipeline variables (avoid using these, as too invisible)
* ~/oj/.gitlab-ci.yml : good place to set variables (but need to commit and push to take effect)
* ~/oj/cxs/cxs.sh : in the "OJ TEST.sh connector scripts"

  * connector scripts based on ~/oj/TEST.sh exist to standardize directory layout and control of diverse other scripts

* ~/o/CSGOptiX/cxs_min.sh : in the underlying script from Opticks (or Opticks code)

  * when using LOCAL envset this has the advantage of needing no commit to change something



Tip for inspecting table content in Grafana web interface
----------------------------------------------------------

Enable "Table view" (slider at top left in dashboard/new) allows to present tables arising from SQL queries::

    select * from opticks_versionset ;
    -- select * from opticks_runs ;
    -- select * from opticks_events ;
    -- SELECT datetime(run_timestamp / 1000000.0, 'unixepoch', 'localtime') AS local_time, total_events AS value FROM opticks_runs;

Keeping a variety of useful "--" commented query snippets in the editor is convenient for development.

* To refresh the table after changing query - just click outside the query editor.



Organizing Grafana SQL queries in a way that avoids duplication between select and where clause
-------------------------------------------------------------------------------------------------

Grafana time series requires (time, value) aliases where time is in seconds,
(for multi-series also select third column aliased as metric).
Two column example::

    SELECT
      r.run_timestamp / 1000000 AS time,
      r.total_events AS value
    FROM
      opticks_runs r
    WHERE
          (r.run_timestamp / 1000000) >= $__from / 1000
      AND (r.run_timestamp / 1000000) <  $__to / 1000
    ORDER BY
      r.run_timestamp ASC;


Note the duplication in the above as where clauses cannot use aliases from
the select clause. The below subquery/common-table-expression CTE approach
can avoid the duplication::

    WITH prepared_data AS (
         SELECT r.run_timestamp / 1000000 AS time, r.total_events AS value
         FROM opticks_runs r
    )
    SELECT time, value FROM prepared_data
    WHERE time >= $__from / 1000 AND time < $__to / 1000
    ORDER BY time ASC;


Restricting time by $__from and $__to millisecond range integrates
with Grafana time picker convention.


Grafana + sqlite perplexing error : "SQL logic error: out of memory (1) "
---------------------------------------------------------------------------

When getting perplexing errors from the Grafana web interface
that do not budge no matter the query.

* go back and press the datasource "TEST" button again
* if it fails check that are using bare database path, not "file:" style


Grafana Web Interface Usage Tips
------------------------------------

Start with Explore view (not Dashboard panel editing) as it is more intuitive:

1. has a "Run Query" button
2. shows table of query results as well as the time series plot


Untested Example of querying with join
------------------------------------------

::

    SELECT
        r.run_timestamp / 1000000 AS time,
        v.nvidia_driver AS "Driver",
        v.opticks_version AS "Opticks Ver",
        r.dt_geometry_upload AS "Upload Duration"
    FROM opticks_runs r
    JOIN opticks_versionset v ON r.versionset_id = v.id
    ORDER BY r.run_timestamp ASC;




Grafana access to URL query parameters
----------------------------------------

To provide a detailed report on a single test in a prior gitlab-ci run I would
need for grafana to accept ci_pipeline_id and ci_job_id in the url query
parameters. How to set that up ? How to access those query prameters and get
them into the SQL queries ?


Step 1: Create Dashboard Variables to Capture URL Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grafana allows you to define variables that automatically "listen" to the
browser's URL query string.

1. Open your Grafana dashboard and click the Dashboard Settings (gear icon) in the top right.

2. Select Variables from the left menu, then click Add variable.

3. Configure the first variable for the Pipeline ID:

   * Name: pipeline_id (This must match what you want to use in your SQL code).
   * Type: Query or Text box (Choose Text box if you just want to pass it natively via URL without querying the DB for values).

::

   select ci_pipeline_id from opticks_runs ;
   select ci_job_id from opticks_runs ;


   * Label: Pipeline ID

4. Repeat the process to create a second variable named job_id as a Text box.

5. Save the dashboard.

💡 The Magic Mechanism: Once a variable is created as a Text box, Grafana
automatically maps URL parameters matching var-variable_name. If your URL
contains ?var-pipeline_id=12345&var-job_id=67890, Grafana will automatically
intercept these values and assign them to your variables.

But they are not independent ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Treating ci_pipeline_id and ci_job_id as independent is an oversimplification.
Is there a better way of handling these togerther when using query parameters ?


Chained variables::

    SELECT DISTINCT ci_pipeline_id FROM opticks_runs ORDER BY run_timestamp DESC;

    SELECT ci_job_id FROM opticks_runs WHERE ci_pipeline_id = '$ci_pipeline_id' ORDER BY run_timestamp DESC;


Combine them::

    SELECT  ci_pipeline_id || ':' || ci_job_id AS value, 'Pipeline #' || ci_pipeline_id || ' -> Job #' || ci_job_id AS text
    FROM opticks_runs ORDER BY run_timestamp DESC LIMIT 20;












Step 2: Use the Variables in Your SQLite Query
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now that Grafana is capturing the parameters, you can reference them in your
SQL code using standard template variable syntax: $pipeline_id and $job_id.

Modify your panel's query to filter directly on these values::


    SELECT
      (r.run_timestamp / 1000000) AS time,
      r.kernel_time_ms AS value,
      r.test_name AS metric
    FROM
      opticks_runs r
    WHERE
      -- Filters rows down to the exact GitLab CI footprint passed in the URL
      r.gitlab_pipeline_id = '$pipeline_id'
      AND r.gitlab_job_id = '$job_id'


* (Note: need the single quotes '$pipeline_id' if your database schema stores
  these IDs as strings/text rather than pure integers).


Step 3: Constructing the Inbound URL (From GitLab or Panels)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To view the targeted dashboard panel report, the incoming URL format must look
exactly like this::

    http://your-grafana-host:3000/d/your-dash-uid/gitlab-test-report?var-pipeline_id=33189&var-job_id=123308



Option A: Generating this link inside a main Grafana Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you have a high-level summary table in Grafana listing recent runs and want
to click a row to dive into a detailed report panel:

1. Edit the Table panel, go to Overrides, and add an override for your pipeline_id column.

2. Add a Data Link property.

3. Set the URL using Grafana's global data variables to pass row details forward dynamically::

    /d/your-dash-uid/gitlab-test-report?var-pipeline_id=${__data.fields.pipeline_id}&var-job_id=${__data.fields.job_id}



Option B: Generating this link from your .gitlab-ci.yml pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want your scheduled pipeline to print out the direct Grafana report URL
right inside the GitLab job log summary, add this to your execution shell
script or after_script:

    after_script:
        - echo "📊 View detailed performance report for this run here:"
        - echo "http://your-grafana-host:3000/d/your-dash-uid/gitlab-test-report?var-pipeline_id=${CI_PIPELINE_ID}&var-job_id=${CI_JOB_ID}"



Grafana Workflow
--------------------

What is the Grafana workflow for getting a query developed in Explore into a
dashboard and how are panels and dashboards organized ? Plus can
"your-dash-uid" be customized ? Also you mentioned the Table panel - whats that ?
How can I find/add one ?


Moving a Query from Explore to a Dashboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once you have successfully dialed in your SQL query inside the Explore
scratchpad, moving it into a dashboard panel takes just a couple of clicks:

1. Look at the top right corner of the Explore view page.

2. Click the "Add to dashboard" button.

3. A dropdown menu will ask if you want to add this query to an Existing
   dashboard or a New dashboard.

4. Once selected, Grafana automatically wraps your query inside a fresh,
   default visualization panel on that dashboard. From there, you can adjust the
   visualization type (e.g., changing it from a Line Graph to a Table).


How Dashboards and Panels Are Organized
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Grafana organizes its user interface using a distinct three-tier hierarchy::

    [ Organizations ]
           └── [ Folders ]
                    └── [ Dashboards ]
                             └── [ Panels & Rows ]

* Organizations: The highest multi-tenant boundary. Most standard setups just
  use a single default organization ("Main Org.").

* Folders: Used to group related dashboards together (e.g., a folder named
  GitLab Performance Testing or Production Infrastructures). You can apply
  security permissions to entire folders.

* Dashboards: A single page workspace containing an layout of graphs,
  variables, and time-pickers.

* Rows & Panels:

   * A Panel is an individual block or card on your dashboard running a single
     query (like a specific line chart, table, or gauge). You can drag, resize, and
     position panels anywhere on the dashboard grid.

    * A Row is a collapsible divider used to cleanly group multiple panels
      together on a busy dashboard page.




Can your-dash-uid Be Customized?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Yes, absolutely. The UID (Unique Identifier) is the string in the URL that
looks like http://localhost:3000/d/abcdef123/my-dashboard.

By default, Grafana randomly generates a 9-character alphanumeric string when
you click Save. However, you can explicitly define your own human-readable UID.
This is incredibly helpful when hardcoding reporting links in your
.gitlab-ci.yml or scripts because it ensures the link never changes if the
dashboard is re-imported.

How to customize it via the UI:

1. Open your dashboard and click the Dashboard Settings (gear icon) at the top right.

2. Under the General tab, look for the Unique Identifier (UID) field.

3. Change it to a clean, slugged identifier (e.g., opticks-ci-reports).

4. Save the dashboard. Your URL will instantly update to /d/opticks-ci-reports/dashboard-name.



How to customize it via Code (Provisioning JSON):

If you are managing your dashboards via JSON files as code (as set up in our
previous step), look at the very top of your exported .json file. Find the
"uid": null or "uid": "abcdef123" property and change it to your desired
string::

    {
      "id": null,
      "uid": "opticks-ci-reports",
      "title": "GitLab Test Report",
      ...
    }


What is the Table Panel & How to Add One?
--------------------------------------------

The Table Panel is one of Grafana’s most powerful native visualization tools.
Instead of turning your numbers into lines, bars, or graphs, it displays raw
query data in a highly structured spreadsheet-like grid.

It is ideal for metadata logging, such as viewing an overview list of recent
gitlab_pipeline_id runs, timestamps, total event counts, and pass/fail metrics.


Key Features of a Table Panel:

* Cell Formatting: You can color code cells based on thresholds (e.g., making a
  cell automatically turn red if a kernel simulation timing goes above 500ms).

* Data Links: As mentioned previously, you can turn any cell or text string
  into a clickable link that opens a deeper, filtered dashboard.


How to Add One:


1. Open your dashboard and click the Add button (the + icon) in the top-right
   toolbar, then select Visualization.

2. A new panel edit interface will open.

3. On the far-right column menu, you will see a dropdown panel titled
   Visualization. Click it (it usually defaults to Time Series).

4. Search for or select Table from the list of visualization icons.

5. Paste your optimized SQLite query into the query editor block below, and
   Grafana will instantly format your rows into a neat table grid!



What is the relationship between datasource and dashboards/panels/folders ?
----------------------------------------------------------------------------

To understand how Grafana connects all these pieces, it helps to look at them
as a decoupled infrastructure stack.

A Data Source lives completely independent of your Dashboards, Panels, and
Folders. Dashboards and panels do not "own" or store your data; they are merely
transparent windows or lenses that look into your data source.

Here is exactly how the relationship works conceptually and structurally.

* Folders (The Filing Cabinet) - organizes
* Dashboards (The Canvas) - single page with collection of variables and time picker

  * A dashboard itself doesn't pull data; it just provides the shared context (like the time range) for the panels sitting inside it.

* Panels (The Lens & The Query)

  * Every individual panel on a dashboard contains an explicit pointer telling it
    which data source to talk to, along with a specific query (your SQL code).

  * When a dashboard loads, every panel independently ships its SQL query down to
    the designated Data Source, pulls the raw tabular rows back up, and instantly
    translates those rows into a chart, gauge, or Table view.



Key Architectural Takeaways
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Mix and Match: A single dashboard is not restricted to one database. You can
  have a dashboard where Panel A pulls simulation timings from your SQLite data
  source, while Panel B right next to it pulls system memory usage metrics from a
  completely different data source (like Prometheus).

* Decoupled Security: If you delete a dashboard or a panel, your SQLite data
  source remains completely untouched. Conversely, if your SQLite data source is
  temporarily offline, your dashboards and panels will still exist perfectly on
  screen—they will simply display a small "Data Source Connection Error" warning
  icon until the database becomes accessible again.



Is is possible to have multiple sqlite databases connected simultaneously organized as separate data sources ?
---------------------------------------------------------------------------------------------------------------

::

    apiVersion: 1

    datasources:
      # Data Source 1: Simulation Benchmarks
      - name: "Opticks Simulation DB"
        type: "frser-sqlite-datasource"
        access: "proxy"
        jsonData:
          path: "/data1/gitlab-runner/sreport_archive/sreportdb.sqlite3"
        editable: false

      # Data Source 2: Hardware Cluster Metrics
      - name: "Workstation Telemetry DB"
        type: "frser-sqlite-datasource"
        access: "proxy"
        jsonData:
          path: "/var/log/metrics/cluster_health.db"
        editable: false



After saving this file and running::

     sudo systemctl restart grafana-server

both options will instantly appear in your system.


If you have multiple identical databases (for example, separate SQLite files
for different test clusters or different projects), you can create a Data
Source Variable in your Dashboard Settings.

1. Create a variable named datasource of type Datasource, and filter it by type
   frser-sqlite-datasource.

2. In your panels, instead of hardcoding a specific database name, set the data
   source to your variable: ${datasource}.

3. This adds a clean dropdown menu to the very top of your dashboard, allowing
   users to instantly flip the entire dashboard's context between Database A,
   Database B, or Database C with a single click.




Realistic Grafana cxs query
-------------------------------


Abuse of time series using the auto metric grouping::

    SELECT
      (e.photon_count * 1.0) / 1000000 AS time, -- Using photon count as the X-Axis scale
      (e.dt_simulate * 1.0) / 1000000 AS value,  -- Using duration as the Y-Axis
      'Run ID: ' || e.run_id AS metric           // This splits the lines automatically
    FROM
      opticks_events e
    WHERE
      e.run_id IN (
        SELECT id FROM opticks_runs ORDER BY run_timestamp DESC LIMIT 5
      )
    ORDER BY
      e.photon_count ASC;



Aggregation to sec_per_photon - better for showing more runs::

    SELECT
      'Run ' || r.id || ' (' || strftime('%m-%d', r.run_timestamp / 1000, 'unixepoch', 'localtime') || ')' AS run_label,
      SUM(e.dt_simulate * 1.0) / SUM(e.photon_count) AS sec_per_photon
    FROM
      opticks_events e
    JOIN
      opticks_runs r ON e.run_id = r.id
    WHERE
      r.id IN (SELECT id FROM opticks_runs ORDER BY run_timestamp DESC LIMIT 10)
    GROUP BY
      r.id
    ORDER BY
      r.run_timestamp ASC;


XY Chart approach::

    SELECT
      (photon_count * 1.0) / 1000000 AS photon_M,
      (dt_simulate * 1.0) / 1000000 AS dt_sim_sec,
      'Run #' || run_id AS run_label
    FROM
      opticks_events
    WHERE
      run_id IN (
        SELECT id FROM opticks_runs ORDER BY run_timestamp DESC LIMIT 5
      )
    ORDER BY
      photon_M ASC;




Try adapt for TargetRun
------------------------

::

    SELECT
      (photon_count * 1.0) / 1000000 AS photon_M,
      (dt_simulate * 1.0) / 1000000 AS dt_sim_sec
    FROM
      opticks_events
    WHERE
      run_id = $TargetRun
    ORDER BY
      photon_M ASC;






External linking from gitlab-ci job outputs to corresponding grafana dashboard
-------------------------------------------------------------------------------

::

    SELECT
      event_index,
      (photon_count * 1.0) / 1000000 AS photon_M,
      (dt_simulate * 1.0) / 1000000 AS dt_sim_sec
    FROM
      opticks_events
    WHERE
      run_id = (
        SELECT id
        FROM opticks_runs
        WHERE ci_pipeline_id = '$ci_pipeline_id' AND ci_job_id = '$ci_job_id'
        LIMIT 1
      )
    ORDER BY
      photon_M ASC;



Annotation Query
-----------------

::

    SELECT
                  r.run_timestamp / 1000000 AS time,
                  'Pipeline #' || r.ci_pipeline_id AS title,
                  'Job ID: ' || r.ci_job_id || ' | Total Events: ' || r.total_events AS text,
                  'gitlab-run,opticks' AS tags
                FROM opticks_runs r
                WHERE r.run_timestamp >= $__from * 1000 AND r.run_timestamp <= $__to * 1000
                ORDER BY r.run_timestamp ASC;


sqlite json_extract to reduce how often need to change tables
----------------------------------------------------------------

::

    -- Example query extracting nested photon source positions and kernel times
    SELECT
        gpu_type,
        opticks_version,
        CAST(json_extract(metrics, '$.photon_counts.total') AS INT) AS total_photons,
        CAST(json_extract(metrics, '$.kernels.raytrace_time') AS REAL) AS raytrace_time
    FROM simulation_runs
    WHERE gpu_type LIKE '%RTX%'
      AND json_extract(metrics, '$.source_position.z') = 0.0;





