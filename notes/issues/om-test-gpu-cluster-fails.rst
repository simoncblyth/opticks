om-test-gpu-cluster-fails
============================


After gdml2gltf, geocache prep and RNG prep down to the familiar 2 fails
---------------------------------------------------------------------------

::

    CTestLog :                 okg4 :      0/     1 : 2019-04-28 20:58:23 : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/okg4/ctest.log 
    CTestLog :                 g4ok :      0/     1 : 2019-04-28 20:58:24 : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/g4ok/ctest.log 
     totals  2   / 385 


    FAILS:
      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          Child aborted***Exception:     3.95   
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   Child aborted***Exception:     3.16   
    rc 0
    gpu019.ihep.ac.cn




fresh install fails
--------------------

::

    FAILS:
      33 /121 Test #33 : NPYTest.NStateTest                            Child aborted***Exception:     0.13       

      ## cannot reproduce the problem

      83 /121 Test #83 : NPYTest.NCSGLoadTest                          Child aborted***Exception:     0.20       
      90 /121 Test #90 : NPYTest.NScanTest                             Child aborted***Exception:     0.16   
      114/121 Test #114: NPYTest.NSceneTest                            Child aborted***Exception:     3.57   
      118/121 Test #118: NPYTest.NSceneMeshTest                        Child aborted***Exception:     2.42   

      ## missing srcnodes.npy file (result of gdml2gltf.py)
      ## fixed after :  op.sh --gdml2gltf    on L7

      10 /50  Test #10 : GGeoTest.GMaterialLibTest                     Child aborted***Exception:     0.15   
      13 /50  Test #13 : GGeoTest.GScintillatorLibTest                 ***Exception: SegFault         0.27   
      16 /50  Test #16 : GGeoTest.GBndLibTest                          Child aborted***Exception:     0.16   
      17 /50  Test #17 : GGeoTest.GBndLibInitTest                      Child aborted***Exception:     0.17   
      30 /50  Test #30 : GGeoTest.GPmtTest                             Child aborted***Exception:     0.15   
      31 /50  Test #31 : GGeoTest.BoundariesNPYTest                    Child aborted***Exception:     0.18   
      32 /50  Test #32 : GGeoTest.GAttrSeqTest                         Child aborted***Exception:     0.39   
      36 /50  Test #36 : GGeoTest.GGeoLibTest                          Child aborted***Exception:     0.18   
      37 /50  Test #37 : GGeoTest.GGeoTest                             Child aborted***Exception:     0.21   
      38 /50  Test #38 : GGeoTest.GMakerTest                           Child aborted***Exception:     0.20   
      45 /50  Test #45 : GGeoTest.GSurfaceLibTest                      Child aborted***Exception:     0.19   
      47 /50  Test #47 : GGeoTest.NLookupTest                          Child aborted***Exception:     0.16   
      48 /50  Test #48 : GGeoTest.RecordsNPYTest                       Child aborted***Exception:     0.18   
      49 /50  Test #49 : GGeoTest.GSceneTest                           Child aborted***Exception:     0.21   

      ### missing geocache (old style)
      ### fixed after :  op.sh -G 


      1  /3   Test #1  : OpticksGeoTest.OpticksGeoTest                 Child aborted***Exception:     0.33   
      2  /3   Test #2  : OpticksGeoTest.OpticksHubTest                 Child aborted***Exception:     0.34   

      ###  fixed following the above changes

      10 /19  Test #10 : OptiXRapTest.OOboundaryTest                   Child aborted***Exception:     0.43   
      11 /19  Test #11 : OptiXRapTest.OOboundaryLookupTest             Child aborted***Exception:     0.49   
      15 /19  Test #15 : OptiXRapTest.OEventTest                       Child aborted***Exception:     0.33   
      16 /19  Test #16 : OptiXRapTest.OInterpolationTest               Child aborted***Exception:     0.42   

      ## these also fixed folloing priming the cache


      17 /19  Test #17 : OptiXRapTest.ORayleighTest                    Child aborted***Exception:     0.59   

      ## fixed after running : opticks-prepare-installcache

      18 /19  Test #18 : OptiXRapTest.intersect_analytic_test          ***Exception: Numerical        12.48  
      19 /19  Test #19 : OptiXRapTest.Roots3And4Test                   ***Exception: Numerical        12.09  

      ## these still failing : familar quartic issue still there with Tesla V100


      1  /5   Test #1  : OKOPTest.OpIndexerTest                        Child aborted***Exception:     0.53   
      2  /5   Test #2  : OKOPTest.OpSeederTest                         Child aborted***Exception:     0.44   
      5  /5   Test #5  : OKOPTest.OpSnapTest                           Child aborted***Exception:     0.36   

      ## now passing 

      2  /5   Test #2  : OKTest.OKTest                                 Child aborted***Exception:     0.42   
      3  /5   Test #3  : OKTest.OTracerTest                            Child aborted***Exception:     0.43   

      ## now passing   

      1  /1   Test #1  : OKG4Test.OKG4Test                             Child aborted***Exception:     0.61   

      ## now passing   


      1  /34  Test #1  : CFG4Test.CMaterialLibTest                     Child aborted***Exception:     0.55   
      2  /34  Test #2  : CFG4Test.CMaterialTest                        Child aborted***Exception:     0.40   
      3  /34  Test #3  : CFG4Test.CTestDetectorTest                    Child aborted***Exception:     0.40   
      5  /34  Test #5  : CFG4Test.CGDMLDetectorTest                    Child aborted***Exception:     0.35   
      6  /34  Test #6  : CFG4Test.CGeometryTest                        Child aborted***Exception:     0.31   
      7  /34  Test #7  : CFG4Test.CG4Test                              Child aborted***Exception:     0.57   
      22 /34  Test #22 : CFG4Test.CGenstepCollectorTest                Child aborted***Exception:     0.40   
      23 /34  Test #23 : CFG4Test.CInterpolationTest                   Child aborted***Exception:     0.39   
      25 /34  Test #25 : CFG4Test.CGROUPVELTest                        Child aborted***Exception:     0.38   
      29 /34  Test #29 : CFG4Test.CRandomEngineTest                    Child aborted***Exception:     0.35   
      32 /34  Test #32 : CFG4Test.CCerenkovGeneratorTest               Child aborted***Exception:     0.43   
      33 /34  Test #33 : CFG4Test.CGenstepSourceTest                   Child aborted***Exception:     0.32   

      




Missing srcnodes.npy
---------------------

::

    NScanTest: /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NPYList.cpp:89: void NPYList::loadBuffer(const char*, int, const char*): Assertion `0' failed.

    Program received signal SIGABRT, Aborted.
    0x00007ffff48a6207 in raise () from /usr/lib64/libc.so.6
    Missing separate debuginfos, use: debuginfo-install boost-filesystem-1.53.0-27.el7.x86_64 boost-program-options-1.53.0-27.el7.x86_64 boost-regex-1.53.0-27.el7.x86_64 boost-system-1.53.0-27.el7.x86_64 glibc-2.17-260.el7.x86_64 keyutils-libs-1.5.8-3.el7.x86_64 krb5-libs-1.15.1-34.el7.x86_64 libcom_err-1.42.9-12.el7_5.x86_64 libgcc-4.8.5-28.el7_5.1.x86_64 libicu-50.1.2-15.el7.x86_64 libselinux-2.5-14.1.el7.x86_64 libstdc++-4.8.5-28.el7_5.1.x86_64 openssl-libs-1.0.2k-16.el7_6.1.x86_64 pcre-8.32-17.el7.x86_64 zlib-1.2.7-17.el7.x86_64
    (gdb) bt
    #0  0x00007ffff48a6207 in raise () from /usr/lib64/libc.so.6
    #1  0x00007ffff48a78f8 in abort () from /usr/lib64/libc.so.6
    #2  0x00007ffff489f026 in __assert_fail_base () from /usr/lib64/libc.so.6
    #3  0x00007ffff489f0d2 in __assert_fail () from /usr/lib64/libc.so.6
    #4  0x00007ffff791da3a in NPYList::loadBuffer (this=0x61d770, treedir=0x61d700 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", bid=0, msg=0x0)
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NPYList.cpp:89
    #5  0x00007ffff79e6fb2 in NCSGData::loadsrc (this=0x61d490, treedir=0x61d700 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248")
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSGData.cpp:81
    #6  0x00007ffff79e0657 in NCSG::loadsrc (this=0x61d650) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSG.cpp:224
    #7  0x00007ffff79dfc65 in NCSG::Load (treedir=0x61cea8 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248", config=0x61cf80)
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSG.cpp:77
    #8  0x00007ffff79df95f in NCSG::Load (treedir=0x61cea8 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248") at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSG.cpp:47
    #9  0x00007ffff79eb2c8 in NCSGList::loadTree (this=0x61bf20, idx=248, boundary=0x622578 "extras/245") at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSGList.cpp:254
    #10 0x00007ffff79eac8f in NCSGList::load (this=0x61bf20) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSGList.cpp:156
    #11 0x00007ffff79ea42f in NCSGList::Load (csgpath=0x61cbc0 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras", verbosity=0, checkmaterial=false)
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSGList.cpp:40
    #12 0x000000000040914c in main (argc=1, argv=0x7fffffffcf18) at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/tests/NScanTest.cc:150
    (gdb) f 5
    #5  0x00007ffff79e6fb2 in NCSGData::loadsrc (this=0x61d490, treedir=0x61d700 "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/extras/248")
        at /afs/ihep.ac.cn/users/b/blyth/g/opticks/npy/NCSGData.cpp:81
    81      m_npy->loadBuffer( treedir,(int)SRC_NODES ); 
    (gdb) 



Missing geocache
------------------

::

    blyth@lxslc702~/g/opticks/ggeo GMaterialLibTest
    2019-04-28 20:19:21.361 ERROR [10721] [OpticksResource::initRunResultsDir@262] /tmp/blyth/opticks/results/GMaterialLibTest/runlabel/20190428_201921
    2019-04-28 20:19:21.365 INFO  [10721] [main@124]  ok 
    2019-04-28 20:19:21.378 WARN  [10721] [NPY<T>::load@659] NPY<T>::load failed for path [/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib/GMaterialLib.npy] use debugload to see why
    2019-04-28 20:19:21.378 FATAL [10721] [GPropertyLib::loadFromCache@571] GPropertyLib::loadFromCache FAILED  dir /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae/GMaterialLib name GMaterialLib.npy
    GMaterialLibTest: /afs/ihep.ac.cn/users/b/blyth/g/opticks/ggeo/GPropertyLib.cc:576: void GPropertyLib::loadFromCache(): Assertion `buf && "YOU PROBABLY NEED TO CREATE/RE-CREATE THE GEOCACHE BY RUNNING  : op.sh -G "' failed.
    Aborted (core dumped)
    blyth@lxslc702~/g/opticks/ggeo 



ORayleighTest fail : missing RNG
------------------------------------

::

    2019-04-28 20:34:29.287 INFO  [261886] [OGeo::convertMergedMesh@264] ) 5 numInstances 672
    2019-04-28 20:34:29.288 INFO  [261886] [OGeo::convert@227] ] nmm 6
    2019-04-28 20:34:29.293 INFO  [261886] [OScene::init@165] ]
    2019-04-28 20:34:29.293 INFO  [261886] [main@59]  ok 
    ORayleighTest: /afs/ihep.ac.cn/users/b/blyth/g/opticks/cudarap/cuRANDWrapper.cc:482: int cuRANDWrapper::LoadIntoHostBuffer(curandState*, unsigned int): Assertion `0' failed.
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    /afs/ihep.ac.cn/users/b/blyth/g/job.sh: line 42: 261886 Aborted                 (core dumped) ORayleighTest
    rc 134
    gpu019.ihep.ac.cn

    

FPE from quartics
---------------------

::

    2019-04-28 20:38:39.706 INFO  [263659] [main@28]  cu_name intersect_analytic_torus_test.cu progname intersect_analytic_torus_test
    2019-04-28 20:38:40.409 INFO  [263659] [main@36]  stack_size 2688
    2019-04-28 20:38:40.410 INFO  [263659] [OptiXTest::init@39] OptiXTest::init cu intersect_analytic_torus_test.cu ptxpath /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/OptiXRap_generated_intersect_analytic_torus_test.cu.ptx raygen intersect_analytic_torus_test exception exception
     cu intersect_analytic_torus_test.cu ptxpath /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/OptiXRap_generated_intersect_analytic_torus_test.cu.ptx raygen intersect_analytic_torus_test exception exception
    2019-04-28 20:38:43.120 INFO  [263659] [OGeo::CreateInputUserBuffer@961] OGeo::CreateInputUserBuffer name planBuffer ctxname intersect_analytic_torus_test src shape 6,4 numBytes 96 elementSize 16 size 6
    /afs/ihep.ac.cn/users/b/blyth/g/job.sh: line 44: 263659 Floating point exception(core dumped) intersect_analytic_test
    rc 136
    gpu019.ihep.ac.cn



Note the huge stack, but there is only one thread::

    CUDA_VISIBLE_DEVICES : 
    gpu019.ihep.ac.cn
    2019-04-28 20:40:08.159 INFO  [264239] [OptiXTest::init@39] OptiXTest::init cu Roots3And4Test.cu ptxpath /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/OptiXRap_generated_Roots3And4Test.cu.ptx raygen Roots3And4Test exception exception
    2019-04-28 20:40:08.164 INFO  [264239] [OptiXTest::Summary@75] Roots3And4Test cu Roots3And4Test.cu ptxpath /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/optixrap/OptiXRap_generated_Roots3And4Test.cu.ptx raygen Roots3And4Test exception exception
    2019-04-28 20:40:10.721 INFO  [264239] [main@32]  stack_size 153728
    /afs/ihep.ac.cn/users/b/blyth/g/job.sh: line 45: 264239 Floating point exception(core dumped) Roots3And4Test
    rc 136
    gpu019.ihep.ac.cn





