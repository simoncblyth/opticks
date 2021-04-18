OpSnapTest_debug_slowdown_with_new_geometry
=============================================


PROBLEM MM 5 (CAUTION UNCONTROLLED MM INDEX IN 5/6/7/8) lvIdx 96  
-------------------------------------------------------------------- 

::

    2021-04-19 02:35:44.248 INFO  [32586] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2021-04-19 02:35:44.248 INFO  [32586] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0xbef4c0
    mm index   0 geocode   A                  numVolumes       3084 numFaces      183096 numITransforms           1 numITransforms*numVolumes        3084 GParts Y GPts Y
    mm index   1 geocode   A                  numVolumes          5 numFaces        1584 numITransforms       25600 numITransforms*numVolumes      128000 GParts Y GPts Y
    mm index   2 geocode   A                  numVolumes          6 numFaces        3504 numITransforms       12612 numITransforms*numVolumes       75672 GParts Y GPts Y
    mm index   3 geocode   A                  numVolumes          6 numFaces        5980 numITransforms        5000 numITransforms*numVolumes       30000 GParts Y GPts Y
    mm index   4 geocode   A                  numVolumes          6 numFaces        3284 numITransforms        2400 numITransforms*numVolumes       14400 GParts Y GPts Y

    mm index   5 geocode   A                  numVolumes          1 numFaces        1272 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y

    mm index   6 geocode   A                  numVolumes          1 numFaces         528 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   7 geocode   A                  numVolumes          1 numFaces         960 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   8 geocode   A                  numVolumes          1 numFaces         384 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   9 geocode   A                  numVolumes        130 numFaces        1560 numITransforms         504 numITransforms*numVolumes       65520 GParts Y GPts Y
     num_remainder_volumes 3084 num_instanced_volumes 315952 num_remainder_volumes + num_instanced_volumes 319036 num_total_faces 202152 num_total_faces_woi 125348744 (woi:without instancing) 
       0 pts Y  GPts.NumPt  3084 lvIdx ( 130 12 11 3 0 1 2 10 9 8 ... 88 88 88 88 88 118 115 116 117)
       1 pts Y  GPts.NumPt     5 lvIdx ( 114 112 110 111 113)
       2 pts Y  GPts.NumPt     6 lvIdx ( 103 98 102 101 99 100)
       3 pts Y  GPts.NumPt     6 lvIdx ( 109 104 108 107 105 106)
       4 pts Y  GPts.NumPt     6 lvIdx ( 126 121 125 124 122 123)

     **5 pts Y  GPts.NumPt     1 lvIdx ( 96)**

       6 pts Y  GPts.NumPt     1 lvIdx ( 93)
       7 pts Y  GPts.NumPt     1 lvIdx ( 94)
       8 pts Y  GPts.NumPt     1 lvIdx ( 95)
       9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
    2021-04-19 02:35:44.249 INFO  [32586] [OGeo::convert@301] [ nmm 10
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
    2021-04-19 02:35:44.249 ERROR [32586] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
    2021-04-19 02:35:44.278 ERROR [32586] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
    2021-04-19 02:35:44.278 ERROR [32586] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
    2021-04-19 02:35:44.279 ERROR [32586] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
    2021-04-19 02:35:44.279 ERROR [32586] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
    2021-04-19 02:35:44.279 INFO  [32586] [OGeo::convert@322] ] nmm 10
    2021-04-19 02:35:44.280 INFO  [32586] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
    2021-04-19 02:35:44.280 INFO  [32586] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix snap-emm-5-
    2021-04-19 02:35:44.280 ERROR [32586] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn 304632 cmdline_target 0 gdmlaux_target -1 active_target 304632
    2021-04-19 02:35:44.281 INFO  [32586] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
     count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/snap-emm-5-00000.jpg dt     1.1119
    2021-04-19 02:35:45.546 INFO  [32586] [OTracer::report@192] OpTracer::snap
     trace_count              1 trace_prep          0.0005 avg     0.0005
     trace_time          1.1774 avg     1.1774

    2021-04-19 02:35:45.547 INFO  [32586] [BTimes::dump@183] OTracer::report
                  validate000                   0.0003
                   compile000                   0.0000
                 prelaunch000                   0.0639
                    launch000                   1.1119
                    launchAVG                   1.1119
    2021-04-19 02:35:45.547 INFO  [32586] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210419_023542
    2021-04-19 02:35:45.547 INFO  [32586] [BFile::preparePath@844] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210419_023542
    2021-04-19 02:35:45.548 INFO  [32586] [OpTracer::snap@182] ]





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


Suspect the problem will be the "temple"  : NOPE THE TEMPLE NOT
--------------------------------------------------------------------

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





geocache-simple
---------------------

::

    O[blyth@localhost opticks]$ geocache-simple()
    > {
    >     local mm
    >     local cmd 
    >     for mm in $(geocache-simple-mm) ; do   
    >         cmd="OpSnapTest --target 304632 --eye -1,-1,-1  --rtx 1 --cvd 1 --enabledmergedmesh $mm --snapoverrideprefix simple-enabledmergedmesh-$mm"
    >         echo $cmd
    >         eval $cmd 
    >     done 
    > }



