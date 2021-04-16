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









geocache-simple
---------------------


O[blyth@localhost opticks]$ geocache-
O[blyth@localhost opticks]$ geocache-simple-mm
0
1
2
3
4
5
6
7
8
9
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
O[blyth@localhost opticks]$ geocache-simple
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 0 --snapoverrideprefix simple-enabledmergedmesh-0
2021-04-17 03:55:51.758 INFO  [293738] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:55:52.896 INFO  [293738] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:55:52.957 INFO  [293738] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:55:53.008 INFO  [293738] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
7 03:55:53.231 INFO  [293738] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-000000.jpg dt     0.0573
2021-04-17 03:55:53.440 INFO  [293738] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0002 avg     0.0002
 trace_time          0.1163 avg     0.1163

2021-04-17 03:55:53.441 INFO  [293738] [BTimes::dump@183] OTracer::report
              validate000                   0.0000
               compile000                   0.0000
             prelaunch000                   0.0581
                launch000                   0.0573
                launchAVG                   0.0573
2021-04-17 03:55:53.441 INFO  [293738] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035551
2021-04-17 03:55:53.441 INFO  [293738] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035551
2021-04-17 03:55:53.441 INFO  [293738] [OpTracer::snap@182] ]



OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 1 --snapoverrideprefix simple-enabledmergedmesh-1
:55:55.750 INFO  [293884] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-100000.jpg dt     0.0023
2021-04-17 03:55:56.266 INFO  [293884] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0346 avg     0.0346
 trace_time          0.4301 avg     0.4301

2021-04-17 03:55:56.266 INFO  [293884] [BTimes::dump@183] OTracer::report
              validate000                   0.0248
               compile000                   0.0000
             prelaunch000                   0.3777
                launch000                   0.0023
                launchAVG                   0.0023
2021-04-17 03:55:56.266 INFO  [293884] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035553
2021-04-17 03:55:56.266 INFO  [293884] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035553
2021-04-17 03:55:56.266 INFO  [293884] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 2 --snapoverrideprefix simple-enabledmergedmesh-2
2021-04-17 03:55:56.433 INFO  [294053] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:55:56.434 INFO  [294053] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:55:56.434 INFO  [294053] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:55:56.435 INFO  [294053] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:55:56.437 INFO  [294053] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:55:56.440 INFO  [294053] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:55:57.486 INFO  [294053] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:55:57.506 INFO  [294053] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:55:57.555 INFO  [294053] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:55:57.600 INFO  [294053] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:55:57.600 INFO  [294053] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:55:57.600 INFO  [294053] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:55:57.600 INFO  [294053] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:55:57.600 INFO  [294053] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:55:57.737 INFO  [294053] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:55:57.737 INFO  [294053] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x94c430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:55:57.737 INFO  [294053] [OGeo::convert@301] [ nmm 10
2021-04-17 03:55:57.737 ERROR [294053] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:55:57.737 ERROR [294053] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:55:58.088 ERROR [294053] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:55:58.088 INFO  [294053] [OGeo::convert@322] ] nmm 10
2021-04-17 03:55:58.110 INFO  [294053] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:55:58.110 INFO  [294053] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-2
2021-04-17 03:55:58.110 ERROR [294053] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:55:58.124 INFO  [294053] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-200000.jpg dt     0.0064
2021-04-17 03:55:58.464 INFO  [294053] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0140 avg     0.0140
 trace_time          0.2362 avg     0.2362

2021-04-17 03:55:58.465 INFO  [294053] [BTimes::dump@183] OTracer::report
              validate000                   0.0119
               compile000                   0.0000
             prelaunch000                   0.2059
                launch000                   0.0064
                launchAVG                   0.0064
2021-04-17 03:55:58.465 INFO  [294053] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035556
2021-04-17 03:55:58.465 INFO  [294053] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035556
2021-04-17 03:55:58.465 INFO  [294053] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 3 --snapoverrideprefix simple-enabledmergedmesh-3
2021-04-17 03:55:58.684 INFO  [294235] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:55:58.685 INFO  [294235] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:55:58.685 INFO  [294235] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:55:58.686 INFO  [294235] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:55:58.688 INFO  [294235] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:55:58.692 INFO  [294235] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:55:59.770 INFO  [294235] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:55:59.791 INFO  [294235] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:55:59.862 INFO  [294235] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:55:59.913 INFO  [294235] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:55:59.913 INFO  [294235] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:55:59.913 INFO  [294235] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:55:59.913 INFO  [294235] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:55:59.913 INFO  [294235] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:00.067 INFO  [294235] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:00.067 INFO  [294235] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0xd7f430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:00.067 INFO  [294235] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:00.067 ERROR [294235] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:00.067 ERROR [294235] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:00.067 ERROR [294235] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:00.215 ERROR [294235] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:00.216 ERROR [294235] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:00.216 ERROR [294235] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:00.216 ERROR [294235] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:00.216 ERROR [294235] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:00.216 ERROR [294235] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:00.216 INFO  [294235] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:00.224 INFO  [294235] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:00.224 INFO  [294235] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-3
2021-04-17 03:56:00.224 ERROR [294235] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:00.229 INFO  [294235] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-300000.jpg dt     0.0072
2021-04-17 03:56:00.472 INFO  [294235] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0048 avg     0.0048
 trace_time          0.1442 avg     0.1442

2021-04-17 03:56:00.472 INFO  [294235] [BTimes::dump@183] OTracer::report
              validate000                   0.0041
               compile000                   0.0000
             prelaunch000                   0.1286
                launch000                   0.0072
                launchAVG                   0.0072
2021-04-17 03:56:00.472 INFO  [294235] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035558
2021-04-17 03:56:00.473 INFO  [294235] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035558
2021-04-17 03:56:00.473 INFO  [294235] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 4 --snapoverrideprefix simple-enabledmergedmesh-4
2021-04-17 03:56:00.597 INFO  [294381] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:00.599 INFO  [294381] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:00.599 INFO  [294381] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:00.599 INFO  [294381] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:00.602 INFO  [294381] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:00.605 INFO  [294381] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:01.644 INFO  [294381] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:01.664 INFO  [294381] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:01.745 INFO  [294381] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:01.797 INFO  [294381] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:01.797 INFO  [294381] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:01.797 INFO  [294381] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:01.797 INFO  [294381] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:01.797 INFO  [294381] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:01.959 INFO  [294381] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:01.959 INFO  [294381] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x25ca430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:01.960 INFO  [294381] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:01.960 ERROR [294381] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:01.960 ERROR [294381] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:01.960 ERROR [294381] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:01.960 ERROR [294381] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:02.039 ERROR [294381] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:02.039 ERROR [294381] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:02.039 ERROR [294381] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:02.039 ERROR [294381] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:02.039 ERROR [294381] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:02.039 INFO  [294381] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:02.044 INFO  [294381] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:02.044 INFO  [294381] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-4
2021-04-17 03:56:02.044 ERROR [294381] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:02.046 INFO  [294381] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-400000.jpg dt     0.0045
2021-04-17 03:56:02.239 INFO  [294381] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0021 avg     0.0021
 trace_time          0.1048 avg     0.1048

2021-04-17 03:56:02.239 INFO  [294381] [BTimes::dump@183] OTracer::report
              validate000                   0.0017
               compile000                   0.0000
             prelaunch000                   0.0962
                launch000                   0.0045
                launchAVG                   0.0045
2021-04-17 03:56:02.239 INFO  [294381] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035600
2021-04-17 03:56:02.239 INFO  [294381] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035600
2021-04-17 03:56:02.240 INFO  [294381] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 5 --snapoverrideprefix simple-enabledmergedmesh-5
2021-04-17 03:56:02.391 INFO  [294519] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:02.392 INFO  [294519] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:02.392 INFO  [294519] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:02.393 INFO  [294519] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:02.395 INFO  [294519] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:02.399 INFO  [294519] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:03.441 INFO  [294519] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:03.459 INFO  [294519] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:03.521 INFO  [294519] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:03.541 INFO  [294519] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:03.541 INFO  [294519] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:03.541 INFO  [294519] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:03.541 INFO  [294519] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:03.541 INFO  [294519] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:03.711 INFO  [294519] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:03.711 INFO  [294519] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x1426430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:03.711 INFO  [294519] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:03.711 ERROR [294519] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:03.711 ERROR [294519] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:03.711 ERROR [294519] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:03.711 ERROR [294519] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:03.711 ERROR [294519] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:03.741 ERROR [294519] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:03.741 ERROR [294519] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:03.741 ERROR [294519] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:03.741 ERROR [294519] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:03.741 INFO  [294519] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:03.743 INFO  [294519] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:03.743 INFO  [294519] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-5
2021-04-17 03:56:03.743 ERROR [294519] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:03.743 INFO  [294519] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-500000.jpg dt     1.1314
2021-04-17 03:56:05.011 INFO  [294519] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0005 avg     0.0005
 trace_time          1.1873 avg     1.1873

2021-04-17 03:56:05.011 INFO  [294519] [BTimes::dump@183] OTracer::report
              validate000                   0.0003
               compile000                   0.0000
             prelaunch000                   0.0545
                launch000                   1.1314
                launchAVG                   1.1314
2021-04-17 03:56:05.011 INFO  [294519] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035602
2021-04-17 03:56:05.012 INFO  [294519] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035602
2021-04-17 03:56:05.012 INFO  [294519] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 6 --snapoverrideprefix simple-enabledmergedmesh-6
2021-04-17 03:56:05.149 INFO  [294701] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:05.151 INFO  [294701] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:05.151 INFO  [294701] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:05.151 INFO  [294701] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:05.154 INFO  [294701] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:05.157 INFO  [294701] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:06.196 INFO  [294701] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:06.216 INFO  [294701] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:06.266 INFO  [294701] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:06.300 INFO  [294701] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:06.300 INFO  [294701] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:06.300 INFO  [294701] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:06.300 INFO  [294701] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:06.300 INFO  [294701] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:06.430 INFO  [294701] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:06.430 INFO  [294701] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x105d430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:06.430 INFO  [294701] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:06.430 ERROR [294701] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:06.431 ERROR [294701] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:06.431 ERROR [294701] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:06.431 ERROR [294701] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:06.431 ERROR [294701] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:06.431 ERROR [294701] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:06.461 ERROR [294701] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:06.461 ERROR [294701] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:06.461 ERROR [294701] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:06.461 INFO  [294701] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:06.462 INFO  [294701] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:06.462 INFO  [294701] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-6
2021-04-17 03:56:06.462 ERROR [294701] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:06.463 INFO  [294701] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-600000.jpg dt     0.0061
2021-04-17 03:56:06.644 INFO  [294701] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0005 avg     0.0005
 trace_time          0.0858 avg     0.0858

2021-04-17 03:56:06.645 INFO  [294701] [BTimes::dump@183] OTracer::report
              validate000                   0.0003
               compile000                   0.0000
             prelaunch000                   0.0783
                launch000                   0.0061
                launchAVG                   0.0061
2021-04-17 03:56:06.645 INFO  [294701] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035605
2021-04-17 03:56:06.645 INFO  [294701] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035605
2021-04-17 03:56:06.645 INFO  [294701] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 7 --snapoverrideprefix simple-enabledmergedmesh-7
2021-04-17 03:56:06.811 INFO  [294854] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:06.813 INFO  [294854] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:06.813 INFO  [294854] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:06.813 INFO  [294854] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:06.816 INFO  [294854] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:06.819 INFO  [294854] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:07.886 INFO  [294854] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:07.908 INFO  [294854] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:07.977 INFO  [294854] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:08.000 INFO  [294854] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:08.000 INFO  [294854] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:08.000 INFO  [294854] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:08.000 INFO  [294854] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:08.000 INFO  [294854] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:08.142 INFO  [294854] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:08.142 INFO  [294854] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x16bc430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:08.142 INFO  [294854] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:08.142 ERROR [294854] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:08.172 ERROR [294854] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:08.172 ERROR [294854] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:08.172 INFO  [294854] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:08.173 INFO  [294854] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:08.173 INFO  [294854] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-7
2021-04-17 03:56:08.174 ERROR [294854] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:08.174 INFO  [294854] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-700000.jpg dt     0.0021
2021-04-17 03:56:08.310 INFO  [294854] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0005 avg     0.0005
 trace_time          0.0564 avg     0.0564

2021-04-17 03:56:08.310 INFO  [294854] [BTimes::dump@183] OTracer::report
              validate000                   0.0003
               compile000                   0.0000
             prelaunch000                   0.0528
                launch000                   0.0021
                launchAVG                   0.0021
2021-04-17 03:56:08.310 INFO  [294854] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035606
2021-04-17 03:56:08.311 INFO  [294854] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035606
2021-04-17 03:56:08.311 INFO  [294854] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 8 --snapoverrideprefix simple-enabledmergedmesh-8
2021-04-17 03:56:08.441 INFO  [294986] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:08.443 INFO  [294986] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:08.443 INFO  [294986] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:08.443 INFO  [294986] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:08.446 INFO  [294986] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:08.449 INFO  [294986] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:09.530 INFO  [294986] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:09.549 INFO  [294986] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:09.610 INFO  [294986] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:09.640 INFO  [294986] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:09.640 INFO  [294986] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:09.640 INFO  [294986] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:09.640 INFO  [294986] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:09.640 INFO  [294986] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:09.785 INFO  [294986] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:09.785 INFO  [294986] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x1cf4430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:09.785 INFO  [294986] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:09.785 ERROR [294986] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:09.815 ERROR [294986] [OGeo::convert@314] MergedMesh 9 IS NOT ENABLED 
2021-04-17 03:56:09.815 INFO  [294986] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:09.817 INFO  [294986] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:09.817 INFO  [294986] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-8
2021-04-17 03:56:09.817 ERROR [294986] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:09.818 INFO  [294986] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-800000.jpg dt     0.0051
2021-04-17 03:56:09.955 INFO  [294986] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0005 avg     0.0005
 trace_time          0.0591 avg     0.0591

2021-04-17 03:56:09.955 INFO  [294986] [BTimes::dump@183] OTracer::report
              validate000                   0.0003
               compile000                   0.0000
             prelaunch000                   0.0525
                launch000                   0.0051
                launchAVG                   0.0051
2021-04-17 03:56:09.955 INFO  [294986] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035608
2021-04-17 03:56:09.955 INFO  [294986] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035608
2021-04-17 03:56:09.955 INFO  [294986] [OpTracer::snap@182] ]
OpSnapTest --target 304632 --eye -1,-1,-1 --rtx 1 --cvd 1 --enabledmergedmesh 9 --snapoverrideprefix simple-enabledmergedmesh-9
2021-04-17 03:56:10.078 INFO  [295139] [BOpticksKey::SetKey@90]  spec DetSim0Svc.X4PhysicalVolume.pWorld.85d8514854333c1a7c3fd50cc91507dc
2021-04-17 03:56:10.079 INFO  [295139] [Opticks::init@441] COMPUTE_MODE forced_compute  hostname localhost.localdomain
2021-04-17 03:56:10.080 INFO  [295139] [Opticks::init@450]  mandatory keyed access to geometry, opticksaux 
2021-04-17 03:56:10.080 INFO  [295139] [Opticks::init@469] OpticksSwitches:WITH_SEED_BUFFER WITH_RECORD WITH_SOURCE WITH_ALIGN_DEV WITH_LOGDOUBLE WITH_KLUDGE_FLAT_ZERO_NOPEEK WITH_SENSORLIB 
2021-04-17 03:56:10.082 INFO  [295139] [Opticks::postconfigure@2582]  setting CUDA_VISIBLE_DEVICES envvar internally to 1
2021-04-17 03:56:10.085 INFO  [295139] [OpticksHub::loadGeometry@284] [ /home/blyth/.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/85d8514854333c1a7c3fd50cc91507dc/1
2021-04-17 03:56:11.150 INFO  [295139] [OpticksHub::loadGeometry@316] ]
2021-04-17 03:56:11.169 INFO  [295139] [OContext::InitRTX@339]  --rtx 1 setting  ON
2021-04-17 03:56:11.243 INFO  [295139] [OContext::CheckDevices@223] 
Device 0                      TITAN RTX ordinal 0 Compute Support: 7 5 Total Memory: 25396445184

2021-04-17 03:56:11.268 INFO  [295139] [CDevice::Dump@245] Visible devices[1:TITAN_RTX]
2021-04-17 03:56:11.268 INFO  [295139] [CDevice::Dump@249] CDevice index 0 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:11.268 INFO  [295139] [CDevice::Dump@245] All devices[0:TITAN_V 1:TITAN_RTX]
2021-04-17 03:56:11.268 INFO  [295139] [CDevice::Dump@249] CDevice index 0 ordinal 0 name TITAN V major 7 minor 0 compute_capability 70 multiProcessorCount 80 totalGlobalMem 12652838912
2021-04-17 03:56:11.268 INFO  [295139] [CDevice::Dump@249] CDevice index 1 ordinal 1 name TITAN RTX major 7 minor 5 compute_capability 75 multiProcessorCount 72 totalGlobalMem 25396445184
2021-04-17 03:56:11.406 INFO  [295139] [OGeo::init@240] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
2021-04-17 03:56:11.407 INFO  [295139] [GGeoLib::dump@385] OGeo::convert GGeoLib numMergedMesh 10 ptr 0x21d5430
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
   5 pts Y  GPts.NumPt     1 lvIdx ( 96)
   6 pts Y  GPts.NumPt     1 lvIdx ( 93)
   7 pts Y  GPts.NumPt     1 lvIdx ( 94)
   8 pts Y  GPts.NumPt     1 lvIdx ( 95)
   9 pts Y  GPts.NumPt   130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 ... 4 5 4 5 4 5 4 5 4)
2021-04-17 03:56:11.407 INFO  [295139] [OGeo::convert@301] [ nmm 10
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 0 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 1 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 2 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 3 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 4 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 5 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 6 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 7 IS NOT ENABLED 
2021-04-17 03:56:11.407 ERROR [295139] [OGeo::convert@314] MergedMesh 8 IS NOT ENABLED 
2021-04-17 03:56:11.436 INFO  [295139] [OGeo::convert@322] ] nmm 10
2021-04-17 03:56:11.438 INFO  [295139] [OpPropagator::snap@130]  dir $TMP/okop/OpSnapTest reldir (null)
2021-04-17 03:56:11.438 INFO  [295139] [OpTracer::snap@156] [ BConfig.cfg [steps=0,ext=.jpg]  ekv 2 eki 3 ekf 6 eks 2 [change .cfg with --snapconfig]  dir $TMP/okop/OpSnapTest reldir (null) snapoverrideprefix simple-enabledmergedmesh-9
2021-04-17 03:56:11.438 ERROR [295139] [OpticksAim::setupCompositionTargetting@176]  cmdline_targetpvn -1 cmdline_target 304632 gdmlaux_target -1 active_target 304632
2021-04-17 03:56:11.439 INFO  [295139] [OTracer::trace_@159]  entry_index 0 trace_count 0 resolution_scale 1 pixeltime_scale 1000 size(1920,1080) ZProj.zw (-1.04082,-694.588) front 0.5774,0.5774,0.5774
 count     1 eyex         -1 eyey         -1 eyez         -1 path /tmp/blyth/opticks/okop/OpSnapTest/simple-enabledmergedmesh-900000.jpg dt     0.0028
2021-04-17 03:56:11.605 INFO  [295139] [OTracer::report@192] OpTracer::snap
 trace_count              1 trace_prep          0.0005 avg     0.0005
 trace_time          0.0572 avg     0.0572

2021-04-17 03:56:11.605 INFO  [295139] [BTimes::dump@183] OTracer::report
              validate000                   0.0003
               compile000                   0.0000
             prelaunch000                   0.0530
                launch000                   0.0028
                launchAVG                   0.0028
2021-04-17 03:56:11.605 INFO  [295139] [OTracer::report@209] save to /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035610
2021-04-17 03:56:11.605 INFO  [295139] [BFile::preparePath@842] created directory /home/blyth/local/opticks/results/OpSnapTest/R1_cvd_1/20210417_035610
2021-04-17 03:56:11.606 INFO  [295139] [OpTracer::snap@182] ]
O[blyth@localhost opticks]$ 



