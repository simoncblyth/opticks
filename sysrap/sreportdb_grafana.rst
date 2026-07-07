sreportdb_grafana.rst
=======================

See also::

    ~/oj/bin/grafana.sh - permissions and datasource provisioning setup
    ~/o/sysrap/sreportdb.sql - schema

To follow how the sreportdb database is created see::

    ~/o/sysrap/sreportdb.h
    ~/o/sysrap/tests/sreportdb.cc
    ~/o/sysrap/tests/sreportdb.sh


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


 
