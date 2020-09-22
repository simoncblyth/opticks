interpolationTest-OScene-OGeo-assert-Darwin
=============================================

* CAUTION : panic tendencies ~/logs/panic-sep22-2020.log

  * better to test on Darwin with a smaller geometry than JUNO 


Probably some fallout from changes 

* https://bitbucket.org/simoncblyth/opticks/commits/e9c799de5f209ecd1675b95d788aad8f81cb8d0d



::

    epsilon:optixrap blyth$ lldb_ interpolationTest 
    (lldb) target create "interpolationTest"
    Current executable set to 'interpolationTest' (x86_64).
    (lldb) r
    Process 87276 launched: '/usr/local/opticks/lib/interpolationTest' (x86_64)
    2020-09-22 14:41:45.108 INFO  [27852302] [BOpticksKey::SetKey@75]  spec OKX4Test.X4PhysicalVolume.lWorld0x338c270_PV.ad026c799f5511ddb91eb379efa84bc4
    2020-09-22 14:41:45.109 INFO  [27852302] [Opticks::init@405] INTEROP_MODE hostname epsilon.local
    2020-09-22 14:41:45.109 INFO  [27852302] [Opticks::init@414]  non-legacy mode : ie mandatory keyed access to geometry, opticksaux 
    2020-09-22 14:41:45.113 INFO  [27852302] [BOpticksResource::setupViaKey@832] 
                 BOpticksKey  :  
          spec (OPTICKS_KEY)  : OKX4Test.X4PhysicalVolume.lWorld0x338c270_PV.ad026c799f5511ddb91eb379efa84bc4
                     exename  : OKX4Test
             current_exename  : interpolationTest
                       class  : X4PhysicalVolume
                     volname  : lWorld0x338c270_PV
                      digest  : ad026c799f5511ddb91eb379efa84bc4
                      idname  : OKX4Test_lWorld0x338c270_PV_g4live
                      idfile  : g4ok.gltf
                      idgdml  : g4ok.gdml
                      layout  : 1

    2020-09-22 14:41:45.113 INFO  [27852302] [Opticks::loadOriginCacheMeta@1849]  cachemetapath /usr/local/opticks/geocache/OKX4Test_lWorld0x338c270_PV_g4live/g4ok_gltf/ad026c799f5511ddb91eb379efa84bc4/1/cachemeta.json
    2020-09-22 14:41:45.113 INFO  [27852302] [NMeta::dump@199] Opticks::loadOriginCacheMeta
    {
        "argline": "/usr/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /usr/local/opticks/tds_ngt_pcnk_sycg.gdml -D --globalinstance ",
        "location": "Opticks::updateCacheMeta",
        "rundate": "20200804_114723",
        "runfolder": "OKX4Test",
        "runlabel": "R0_cvd_",
        "runstamp": 1596538043
    }
    2020-09-22 14:41:45.114 INFO  [27852302] [Opticks::loadOriginCacheMeta@1853]  gdmlpath /usr/local/opticks/tds_ngt_pcnk_sycg.gdml
    2020-09-22 14:41:45.114 INFO  [27852302] [OpticksHub::loadGeometry@542] [ /usr/local/opticks/geocache/OKX4Test_lWorld0x338c270_PV_g4live/g4ok_gltf/ad026c799f5511ddb91eb379efa84bc4/1
    2020-09-22 14:41:46.260 INFO  [27852302] [NMeta::dump@199] GGeo::loadCacheMeta.lv2sd
    2020-09-22 14:41:46.260 INFO  [27852302] [NMeta::dump@199] GGeo::loadCacheMeta.lv2mt
    2020-09-22 14:41:46.265 ERROR [27852302] [Opticks::setupTimeDomain@2539]  animtimerange -1.0000,-1.0000,0.0000,0.0000
    2020-09-22 14:41:46.265 INFO  [27852302] [Opticks::setupTimeDomain@2550]  cfg.getTimeMaxThumb [--timemaxthumb] 6 cfg.getAnimTimeMax [--animtimemax] -1 cfg.getAnimTimeMax [--animtimemax] -1 speed_of_light (mm/ns) 300 extent (mm) 60000 rule_of_thumb_timemax (ns) 1200 u_timemax 1200 u_animtimemax 1200
    2020-09-22 14:41:46.266 FATAL [27852302] [Opticks::setProfileDir@546]  dir /tmp/blyth/opticks/interpolationTest/evt/g4live/torch
    2020-09-22 14:41:46.267 INFO  [27852302] [OpticksHub::loadGeometry@586] ]
    2020-09-22 14:41:46.305 ERROR [27852302] [*OpticksGen::makeLegacyGensteps@227]  code 5 srctype torch
    2020-09-22 14:41:46.305 INFO  [27852302] [*Opticks::makeSimpleTorchStep@3335]  enable : --torch (the default)  configure : --torchconfig [NULL] dump details : --torchdbg 
    2020-09-22 14:41:46.305 ERROR [27852302] [*OpticksGen::makeTorchstep@404]  as torchstep isDefault replacing placeholder frame  frameIdx : 0 detectorDefaultFrame : 0
    2020-09-22 14:41:46.306 INFO  [27852302] [OpticksGen::targetGenstep@336] setting frame 0 Id
    2020-09-22 14:41:46.306 ERROR [27852302] [*OpticksGen::makeTorchstep@428]  generateoverride 0 num_photons0 10000 num_photons 10000
    2020-09-22 14:41:46.306 ERROR [27852302] [OContext::SetupOptiXCachePathEnvvar@286] envvar OPTIX_CACHE_PATH not defined setting it internally to /var/tmp/blyth/OptiXCache
    2020-09-22 14:41:47.319 INFO  [27852302] [OContext::CheckDevices@207] 
    Device 0                GeForce GT 750M ordinal 0 Compute Support: 3 0 Total Memory: 2147024896

    2020-09-22 14:41:47.327 INFO  [27852302] [CDevice::Dump@230] Visible devices[0:GeForce_GT_750M]
    2020-09-22 14:41:47.327 INFO  [27852302] [CDevice::Dump@234] CDevice index 0 ordinal 0 name GeForce GT 750M major 3 minor 0 compute_capability 30 multiProcessorCount 2 totalGlobalMem 2147024896
    2020-09-22 14:41:47.327 INFO  [27852302] [CDevice::Dump@230] All devices[0:GeForce_GT_750M]
    2020-09-22 14:41:47.327 INFO  [27852302] [CDevice::Dump@234] CDevice index 0 ordinal 0 name GeForce GT 750M major 3 minor 0 compute_capability 30 multiProcessorCount 2 totalGlobalMem 2147024896
    2020-09-22 14:41:47.327 INFO  [27852302] [OScene::init@119] [
    2020-09-22 14:41:47.363 INFO  [27852302] [OGeo::init@238] OGeo  top Sbvh ggg Sbvh assembly Sbvh instance Sbvh
    2020-09-22 14:41:47.363 INFO  [27852302] [GGeoLib::dump@366] OGeo::convert GGeoLib TRIANGULATED  numMergedMesh 10 ptr 0x109868540
    mm index   0 geocode   T                  numVolumes     316326 numFaces       50136 numITransforms           1 numITransforms*numVolumes      316326 GParts Y GPts Y
    mm index   1 geocode   T                  numVolumes          5 numFaces        1584 numITransforms       25600 numITransforms*numVolumes      128000 GParts Y GPts Y
    mm index   2 geocode   T                  numVolumes          6 numFaces        3504 numITransforms       12612 numITransforms*numVolumes       75672 GParts Y GPts Y
    mm index   3 geocode   T                  numVolumes          6 numFaces        5980 numITransforms        5000 numITransforms*numVolumes       30000 GParts Y GPts Y
    mm index   4 geocode   T                  numVolumes          6 numFaces        3284 numITransforms        2400 numITransforms*numVolumes       14400 GParts Y GPts Y
    mm index   5 geocode   T                  numVolumes          1 numFaces         192 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   6 geocode   T                  numVolumes          1 numFaces         960 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   7 geocode   T                  numVolumes          1 numFaces         384 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   8 geocode   T                  numVolumes          1 numFaces        1272 numITransforms         590 numITransforms*numVolumes         590 GParts Y GPts Y
    mm index   9 geocode   T                  numVolumes        130 numFaces        1560 numITransforms         504 numITransforms*numVolumes       65520 GParts Y GPts Y
     num_total_volumes 316326 num_instanced_volumes 315952 num_global_volumes 374 num_total_faces 68856 num_total_faces_woi 125017544 (woi:without instancing) 
       0 pts Y  GPts.NumPt 374 lvIdx ( 56 12 11 3 0 1 2 10 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 9 8 8 55 54 53 46 45 18 17 13 13 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 14 15 15 15 15 15 15 15 15 16 16 16 16 16 16 16 16 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23  16 16 16 16 16 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 23 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 16 44 41 42 43)
       1 pts Y  GPts.NumPt 5 lvIdx ( 40 38 36 37 39)
       2 pts Y  GPts.NumPt 6 lvIdx ( 29 24 28 27 25 26)
       3 pts Y  GPts.NumPt 6 lvIdx ( 35 30 34 33 31 32)
       4 pts Y  GPts.NumPt 6 lvIdx ( 52 47 51 50 48 49)
       5 pts Y  GPts.NumPt 1 lvIdx ( 19)
       6 pts Y  GPts.NumPt 1 lvIdx ( 20)
       7 pts Y  GPts.NumPt 1 lvIdx ( 21)
       8 pts Y  GPts.NumPt 1 lvIdx ( 22)
       9 pts Y  GPts.NumPt 130 lvIdx ( 7 6 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4 5 4)
    2020-09-22 14:41:47.363 INFO  [27852302] [OGeo::convert@264] [ nmm 10
    2020-09-22 14:41:47.390 FATAL [27852302] [*GMesh::makeFaceRepeatedInstancedIdentityBuffer@2108]  iidentity_ok 0 iidentity_buffer_items 1 numFaces (sum of faces in numVolumes)50136 numITransforms 1 numVolumes*numITransforms 316326 numInstanceIdentity 374 numRepeatedIdentity 50136 m_iidentity_buffer 1,374,4 m_itransforms_buffer 1,4,4
    Assertion failed: (iidentity_ok), function makeFaceRepeatedInstancedIdentityBuffer, file /Users/blyth/opticks/ggeo/GMesh.cc, line 2121.
    Process 87276 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff67e8ab66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff67e8ab66 <+10>: jae    0x7fff67e8ab70            ; <+20>
        0x7fff67e8ab68 <+12>: movq   %rax, %rdi
        0x7fff67e8ab6b <+15>: jmp    0x7fff67e81ae9            ; cerror_nocancel
        0x7fff67e8ab70 <+20>: retq   
    Target 0: (interpolationTest) stopped.
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff67e8ab66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff68055080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff67de61ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff67dae1ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x0000000104248da9 libGGeo.dylib`GMesh::makeFaceRepeatedInstancedIdentityBuffer(this=0x0000000109875e40) at GMesh.cc:2121
        frame #5: 0x0000000104249a6b libGGeo.dylib`GMesh::getFaceRepeatedInstancedIdentityBuffer(this=0x0000000109875e40) at GMesh.cc:2230
        frame #6: 0x00000001042495fa libGGeo.dylib`GMesh::getAppropriateRepeatedIdentityBuffer(this=0x0000000109875e40) at GMesh.cc:2207
        frame #7: 0x000000010012ebab libOptiXRap.dylib`OGeo::makeTriangulatedGeometry(this=0x00000001163ee380, mm=0x0000000109875e40, lod=0) at OGeo.cc:932
        frame #8: 0x000000010012cfc5 libOptiXRap.dylib`OGeo::makeOGeometry(this=0x00000001163ee380, mergedmesh=0x0000000109875e40, lod=0) at OGeo.cc:595
        frame #9: 0x000000010012b5af libOptiXRap.dylib`OGeo::makeGlobalGeometryGroup(this=0x00000001163ee380, mm=0x0000000109875e40) at OGeo.cc:324
        frame #10: 0x000000010012a655 libOptiXRap.dylib`OGeo::convertMergedMesh(this=0x00000001163ee380, i=0) at OGeo.cc:303
        frame #11: 0x0000000100129fdd libOptiXRap.dylib`OGeo::convert(this=0x00000001163ee380) at OGeo.cc:270
        frame #12: 0x000000010011ff29 libOptiXRap.dylib`OScene::init(this=0x00007ffeefbfe548) at OScene.cc:169
        frame #13: 0x000000010011f2e1 libOptiXRap.dylib`OScene::OScene(this=0x00007ffeefbfe548, hub=0x00007ffeefbfe5b0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:91
        frame #14: 0x00000001001204fd libOptiXRap.dylib`OScene::OScene(this=0x00007ffeefbfe548, hub=0x00007ffeefbfe5b0, cmake_target="OptiXRap", ptxrel=0x0000000000000000) at OScene.cc:90
        frame #15: 0x000000010000ba4e interpolationTest`main(argc=1, argv=0x00007ffeefbfe848) at interpolationTest.cc:187
        frame #16: 0x00007fff67d3a015 libdyld.dylib`start + 1
        frame #17: 0x00007fff67d3a015 libdyld.dylib`start + 1
    (lldb) f 15
    frame #15: 0x000000010000ba4e interpolationTest`main(argc=1, argv=0x00007ffeefbfe848) at interpolationTest.cc:187
       184 	
       185 	    Opticks ok(argc, argv);
       186 	    OpticksHub hub(&ok);
    -> 187 	    OScene sc(&hub);
       188 	
       189 	    LOG(info) << " ok " ; 
       190 	
    (lldb) 
      [Restored Sep 22, 2020 at 2:45:24 PM]
    Last login: Tue Sep 22 14:45:24 on ttys002
    .bashrc OPTICKS_MODE dev



     OGeo=INFO GMesh=INFO interpolationTest



