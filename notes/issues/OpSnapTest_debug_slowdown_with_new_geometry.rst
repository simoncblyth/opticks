OpSnapTest_debug_slowdown_with_new_geometry
=============================================


Issue: 2021 April : new geometry timings much lower ? Whats causing the slowdown ?
--------------------------------------------------------------------------------------

::

    OpSnapTest --xanalytic --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 


::

    O[blyth@localhost opticks]$ UseOptiX --uniqrec
    TITAN_V/0
    TITAN_RTX/1


* is --xanalytic still needed ?
* --enabledmergedmesh seems not working ?

::

    OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh 1


    2021-04-17 02:39:48.003 INFO  [157145] [BTimes::dump@183] OTracer::report
                  validate000                   0.0251
                   compile000                   0.0000
                 prelaunch000                   1.2260
                    launch000                   0.0023
                    launchAVG                   0.0023
    2021-04-17 02:39:48.003 INFO  [157145] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_023944
    2021-04-17 02:39:48.003 INFO  [157145] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_023944
    2021-04-17 02:39:48.004 INFO  [157145] [OpTracer::snap@180] ]
    O[blyth@localhost optixrap]$ 



geocache-simple-mm(){ ls -1 $(geocache-keydir)/GMergedMesh ; }
geocache-simple()
{
    local mm
    local cmd 
    for mm in $(geocache-simple-mm) ; do 
        cmd="OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh $mm"
        echo $cmd
    done 
}



