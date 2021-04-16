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


Suspect the problem will be the "temple"
-------------------------------------------

* warning the "5/" is before pinning down repeat_candidate ordering with the two-level-sort 


::

    O[blyth@localhost opticks]$ python3 ana/ggeo.py 5/
    nidx:70258 triplet:5000000 sh:600010 sidx:    0   nrpo( 70258     5     0     0 )  shape(  96  16                              uni_acrylic3                          Water///Acrylic) 

    gt : gg.all_volume_transforms[70258]
    [[   -0.585    -0.805     0.098     0.   ]
     [   -0.809     0.588     0.        0.   ]
     [   -0.057    -0.079    -0.995     0.   ]
     [ 1022.116  1406.822 17734.953     1.   ]]

    tr : transform
    [[   -0.585    -0.805     0.098     0.   ]
     [   -0.809     0.588     0.        0.   ]
     [   -0.057    -0.079    -0.995     0.   ]
     [ 1022.116  1406.822 17734.953     1.   ]]

    it : inverted transform
    [[   -0.585    -0.809    -0.057     0.   ]
     [   -0.805     0.588    -0.079     0.   ]
     [    0.098    -0.       -0.995     0.   ]
     [   -0.       -0.    17820.        1.   ]]

    bb : bbox4
    [[  574.885   960.342 17685.367     1.   ]
     [ 1469.02   1852.852 17893.8       1.   ]]

    cbb : (bb[0]+bb[1])/2.
    [ 1021.952  1406.597 17789.584     1.   ]

    c4 : center4
    [ 1021.952  1406.597 17789.584     1.   ]

    ce : center_extent
    [ 1021.952  1406.597 17789.584   447.067]

    ic4 : np.dot( c4, it) : inverse transform applied to center4 : expect close to origin 
    [  5.608  -0.    -54.344   1.   ]

    ibb : np.dot( bb, it) : inverse transform applied to bbox4 : expect symmetric around origin
    [[ 616.268   99.383  110.248    1.   ]
     [-605.053  -99.383 -218.936    1.   ]]





