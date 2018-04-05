Epsilon (10.13.4 Xcode 9.2 OptiX 5.0.1 CUDA 9.1) 23 fails out of 300 : now 2/300 fails
=========================================================================================

CURRENT STATUS 2/300 fails
-----------------------------

::

    99% tests passed, 2 tests failed out of 300

    Total Test time (real) = 160.34 sec

    The following tests FAILED:
        223 - GGeoTest.GSceneTest (Child aborted)         ## lack of some analytic geocache files 
        267 - OptiXRapTest.OInterpolationTest (Failed)    ## lack of numpy 
    Errors while running CTest
    Thu Apr  5 11:24:23 CST 2018


FIX : add *opticks-check-installcache* prior to *opticks-t* or *subproj-t* running 
-------------------------------------------------------------------------------------

::

    epsilon:~ blyth$ t opticks-full
    opticks-full () 
    { 
        local msg="=== $FUNCNAME :";
        echo $msg START $(date);
        opticks-info;
        if [ ! -d "$(opticks-prefix)/externals" ]; then
            echo $msg installing the below externals into $(opticks-prefix)/externals;
            opticks-externals;
            opticks-externals-install;
        else
            echo $msg using preexisting externals from $(opticks-prefix)/externals;
        fi;
        opticks-configure;
        opticks--;
        opticks-prepare-installcache;
        echo $msg DONE $(date)
    }
    epsilon:~ blyth$ 


Typically *opticks-full* building does not proceed smoothly so *opticks-prepare-installcache* 
ends up not being run before *opticks-t* is used.

Added simple fix at bash level *opticks-check-installcache*::


    epsilon:optickscore blyth$ opticks-;opticks-t
    === opticks-check-installcache : /usr/local/opticks/installcache : missing RNG
    === opticks-check-installcache : /usr/local/opticks/installcache : missing OKC
    === opticks-t- : ABORT : missing installcache components : create with opticks-prepare-installcache
    epsilon:optickscore blyth$ 
    epsilon:optickscore blyth$ 

    epsilon:optickscore blyth$ opticks-prepare-installcache
    === opticks-prepare-installcache : generating RNG seeds into installcache
    2018-04-05 11:01:29.964 INFO  [471395] [main@35]  work 3000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /usr/local/opticks/installcache/RNG
    cuRANDWrapper::Allocate
    cuRANDWrapper::InitFromCacheIfPossible
    cuRANDWrapper::InitFromCacheIfPossible : no cache initing and saving 
    cuRANDWrapper::Init
     init_rng_wrapper sequence_index   0  thread_offset       0  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    20.7649 ms 
     init_rng_wrapper sequence_index   1  thread_offset   32768  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    42.5015 ms 
     init_rng_wrapper sequence_index   2  thread_offset   65536  threads_per_launch  32768 blocks_per_launch    128   threads_per_block    256  kernel_time    59.0853 ms 
     init_rng_wrapper sequence_index   3  thread_offset   98304  threads_per

    ...


::

    The following tests FAILED:
        223 - GGeoTest.GSceneTest (Child aborted)
              ## analytic related 

        265 - OptiXRapTest.bufferTest (Child aborted)
        266 - OptiXRapTest.OEventTest (Child aborted)
              ## unexpected OptiX version

        267 - OptiXRapTest.OInterpolationTest (Failed)
              ## lack of numpy 

    Errors while running CTest
    Thu Apr  5 11:07:48 CST 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/ctest.log


TODO: add opticks-check-geocache to avoid tests needing geocache running without it
---------------------------------------------------------------------------------------






opticks-t first run with no geocache preparation
-------------------------------------------------

First run of opticks-t starting fresh with no geocache gives lots of ggeo-t fails::

    92% tests passed, 23 tests failed out of 300

    Total Test time (real) = 116.51 sec

    The following tests FAILED:

        184 - GGeoTest.GMaterialLibTest (Child aborted)
        187 - GGeoTest.GScintillatorLibTest (Child aborted)
        190 - GGeoTest.GBndLibTest (Child aborted)
        191 - GGeoTest.GBndLibInitTest (Child aborted)
        202 - GGeoTest.GPartsTest (Child aborted)
        204 - GGeoTest.GPmtTest (Child aborted)
        205 - GGeoTest.BoundariesNPYTest (Child aborted)
        206 - GGeoTest.GAttrSeqTest (Child aborted)
        210 - GGeoTest.GGeoLibTest (Child aborted)
        211 - GGeoTest.GGeoTest (Child aborted)
        212 - GGeoTest.GMakerTest (Child aborted)
        219 - GGeoTest.GSurfaceLibTest (Child aborted)
        221 - GGeoTest.NLookupTest (Child aborted)
        222 - GGeoTest.RecordsNPYTest (Child aborted)
        223 - GGeoTest.GSceneTest (Child aborted)
        224 - GGeoTest.GMeshLibTest (Child aborted)

        265 - OptiXRapTest.bufferTest (Child aborted)
        266 - OptiXRapTest.OEventTest (Child aborted)
        267 - OptiXRapTest.OInterpolationTest (Failed)
        268 - OptiXRapTest.ORayleighTest (Child aborted)

        272 - OKOPTest.OpSeederTest (Child aborted)

        277 - OKTest.OKTest (Child aborted)

        300 - okg4Test.OKG4Test (Child aborted)

    Errors while running CTest
    Wed Apr  4 22:01:13 CST 2018
    epsilon:~ blyth$ 


subsequent ggeo-t with no explicit cache creation gives only one fail
------------------------------------------------------------------------

But subsequently running ggeo-t gives only one fail

     48 - GGeoTest.GSceneTest (Child aborted)

* presumably a subsequent higher level opticks-t test 
  creates the geocache, so subsequent ggeo-t succeeds much more

  * TODO: geocache should be populated by the install process ? 
    or just somehow reorder the tests

   
GSceneTest fail for lack of MeshIndexAnalytic
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/PVNames.txt
    DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/LVNames.txt
    DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexSource.json
    DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexLocal.json


::

    2018-04-05 10:02:44.757 INFO  [435229] [GGeoLib::loadConstituents@168] /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1
    2018-04-05 10:02:44.758 INFO  [435229] [GGeoLib::loadConstituents@217] GGeoLib::loadConstituents loaded 0 ridx ()
    2018-04-05 10:02:44.758 WARN  [435229] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/PVNames.txt
    2018-04-05 10:02:44.758 WARN  [435229] [GItemList::load_@66] GItemList::load_ NO SUCH TXTPATH /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/GNodeLibAnalytic/LVNames.txt
    2018-04-05 10:02:44.758 WARN  [435229] [*Index::load@426] Index::load FAILED to load index  idpath /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 itemtype GItemIndex Source path /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexSource.json Local path /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1/MeshIndexAnalytic/GItemIndexLocal.json
    2018-04-05 10:02:44.758 WARN  [435229] [GItemIndex::loadIndex@176] GItemIndex::loadIndex failed for  idpath /usr/local/opticks/geocache/DayaBay_VGDX_20140414-1300/g4_00.dae/96ff965744a2f6b78c24e33c80d3a4cd/1 reldir MeshIndexAnalytic override NULL
    2018-04-05 10:02:44.758 FATAL [435229] [GMeshLib::loadFromCache@61]  meshindex load failure 
    Assertion failed: (has_index && " MISSING MESH INDEX : PERHAPS YOU NEED TO CREATE/RE-CREATE GEOCACHE WITH : op.sh -G "), function loadFromCache, file /Users/blyth/opticks/ggeo/GMeshLib.cc, line 62.
    Abort trap: 6
    epsilon:~ blyth$ 
    epsilon:~ blyth$ 


subsequent oxrap-t stays at 4 fails
--------------------------------------

::

    The following tests FAILED:
         13 - OptiXRapTest.bufferTest (Child aborted)
         14 - OptiXRapTest.OEventTest (Child aborted)
         15 - OptiXRapTest.OInterpolationTest (Failed)
         16 - OptiXRapTest.ORayleighTest (Child aborted)
    Errors while running CTest


bufferTest, OEventTest : unexpected OptiX version
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    epsilon:~ blyth$ bufferTest 
    2018-04-05 10:21:55.556 INFO  [445753] [Opticks::dumpArgs@1105] Opticks::configure argc 2
      0 : bufferTest
      1 : --compute
    2018-04-05 10:21:55.557 INFO  [445753] [main@110] bufferTest OPTIX_VERSION 50001
    Assertion failed: (0 && "unexpected OPTIX_VERSION"), function DefaultWithTop, file /Users/blyth/opticks/optixrap/OConfig.cc, line 46.
    Abort trap: 6

    epsilon:~ blyth$ OEventTest 
    2018-04-05 10:25:39.488 INFO  [449350] [Opticks::dumpArgs@1105] Opticks::configure argc 3
      0 : OEventTest
      1 : --machinery
      2 : --compute
    2018-04-05 10:25:39.727 INFO  [449350] [main@47] OEventTest OPTIX_VERSION 50001
    Assertion failed: (0 && "unexpected OPTIX_VERSION"), function DefaultWithTop, file /Users/blyth/opticks/optixrap/OConfig.cc, line 46.
    Abort trap: 6



OInterpolationTest : lack of numpy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-04-05 10:22:41.335 INFO  [447323] [OContext::close@246] OContext::close m_cfg->apply() done.
    Traceback (most recent call last):
      File "/Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py", line 3, in <module>
        import os,sys, numpy as np, logging
    ImportError: No module named numpy
    2018-04-05 10:22:44.374 INFO  [447323] [SSys::run@50] python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py rc_raw : 256 rc : 1
    2018-04-05 10:22:44.374 WARN  [447323] [SSys::run@57] SSys::run FAILED with  cmd python /Users/blyth/opticks/optixrap/tests/OInterpolationTest_interpol.py
    epsilon:~ blyth$ 


ORayleighTest : missing RNG cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-04-05 10:23:47.557 INFO  [448233] [main@69]  ok 
    2018-04-05 10:23:47.557 ERROR [448233] [ORng::init@41] ORng::init rng_max 3000000 rngCacheDir /usr/local/opticks/installcache/RNG num_mask 0
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 479.
    Abort trap: 6
    epsilon:~ blyth$ 



subsequent okop-t stays at 1 fail
-----------------------------------

::

   The following tests FAILED:
       2 - OKOPTest.OpSeederTest (Child aborted)


OpSeederTest : missing RNG cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    2018-04-05 10:28:38.569 ERROR [452188] [ORng::init@41] ORng::init rng_max 3000000 rngCacheDir /usr/local/opticks/installcache/RNG num_mask 0
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 479.
    Abort trap: 6
    epsilon:~ blyth$ 
     


ok-t
------

OKTest when run via ctest::

    Application Specific Information:
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 479.
     

    Thread 0 Crashed:: Dispatch queue: com.apple.main-thread
    0   libsystem_kernel.dylib        	0x00007fff6a5fab6e __pthread_kill + 10
    1   libsystem_pthread.dylib       	0x00007fff6a7c5080 pthread_kill + 333
    2   libsystem_c.dylib             	0x00007fff6a5561ae abort + 127
    3   libsystem_c.dylib             	0x00007fff6a51e1ac __assert_rtn + 320
    4   libCUDARap.dylib              	0x000000010601bd47 cuRANDWrapper::LoadIntoHostBuffer(curandStateXORWOW*, unsigned int) + 535
    5   libOptiXRap.dylib             	0x0000000108446725 ORng::init() + 1157 (ORng.cc:64)
    6   libOptiXRap.dylib             	0x0000000108446261 ORng::ORng(Opticks*, OContext*) + 129 (ORng.cc:24)
    7   libOptiXRap.dylib             	0x0000000108446915 ORng::ORng(Opticks*, OContext*) + 37 (ORng.cc:25)
    8   libOptiXRap.dylib             	0x00000001084444e0 OPropagator::OPropagator(OpticksHub*, OEvent*, OpticksEntry*) + 320 (OPropagator.cc:65)
    9   libOptiXRap.dylib             	0x000000010844464d OPropagator::OPropagator(OpticksHub*, OEvent*, OpticksEntry*) + 45 (OPropagator.cc:77)
    10  libOKOP.dylib                 	0x000000010880b747 OpEngine::initPropagation() + 183 (OpEngine.cc:80)
    11  libOKOP.dylib                 	0x000000010880b4e2 OpEngine::init() + 802 (OpEngine.cc:71)
    12  libOKOP.dylib                 	0x000000010880b174 OpEngine::OpEngine(OpticksHub*) + 276 (OpEngine.cc:53)
    13  libOKOP.dylib                 	0x000000010880b5fd OpEngine::OpEngine(OpticksHub*) + 29 (OpEngine.cc:55)
    14  libOK.dylib                   	0x00000001088db9c4 OKPropagator::OKPropagator(OpticksHub*, OpticksIdx*, OpticksViz*) + 196 (OKPropagator.cc:46)
    15  libOK.dylib                   	0x00000001088dbb1d OKPropagator::OKPropagator(OpticksHub*, OpticksIdx*, OpticksViz*) + 45 (OKPropagator.cc:52)
    16  libOK.dylib                   	0x00000001088db30e OKMgr::OKMgr(int, char**, char const*) + 654 (OKMgr.cc:50)
    17  libOK.dylib                   	0x00000001088db5bb OKMgr::OKMgr(int, char**, char const*) + 43 (OKMgr.cc:55)
    18  OKTest                        	0x0000000103e8b031 main + 1361 (OKTest.cc:59)


OKTest::

    2018-04-05 10:31:23.242 INFO  [455100] [SLog::operator@15] OEvent::OEvent DONE
    2018-04-05 10:31:23.242 ERROR [455100] [ORng::init@41] ORng::init rng_max 3000000 rngCacheDir /usr/local/opticks/installcache/RNG num_mask 0
    2018-04-05 10:31:23.242 ERROR [455100] [*cuRANDWrapper::instanciate@28] cuRANDWrapper::instanciate num_items 3000000
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 479.
    Abort trap: 6
    epsilon:~ blyth$ 

OKG4Test::

    2018-04-05 10:32:58.280 INFO  [457613] [SLog::operator@15] OEvent::OEvent DONE
    2018-04-05 10:32:58.280 ERROR [457613] [ORng::init@41] ORng::init rng_max 3000000 rngCacheDir /usr/local/opticks/installcache/RNG num_mask 0
    2018-04-05 10:32:58.280 ERROR [457613] [*cuRANDWrapper::instanciate@28] cuRANDWrapper::instanciate num_items 3000000
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /usr/local/opticks/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 479.
    Abort trap: 6


