Greenfield Installation Testing
==================================

opticks-t fails
-----------------

Greenfield::

    89% tests passed, 18 tests failed out of 162

    Total Test time (real) =  85.76 sec

    The following tests FAILED:
          4 - SysRapTest.SSysTest (Failed)
         34 - NPYTest.NOpenMeshCfgTest (OTHER_FAULT)
         35 - NPYTest.NOpenMeshFindTest (OTHER_FAULT)
         80 - GGeoTest.GPartsTest (OTHER_FAULT)
         81 - GGeoTest.GPmtTest (OTHER_FAULT)
         89 - GGeoTest.GGeoTestTest (OTHER_FAULT)
         90 - GGeoTest.GMakerTest (SEGFAULT)
         96 - GGeoTest.GPropertyTest (OTHER_FAULT)
        102 - GGeoTest.GSceneTest (OTHER_FAULT)
        136 - OptiXRapTest.OEventTest (OTHER_FAULT)
        137 - OptiXRapTest.OInterpolationTest (Failed)
        141 - OKOPTest.OpSeederTest (OTHER_FAULT)
        148 - OKTest.VizTest (OTHER_FAULT)
        150 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        151 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
        154 - cfg4Test.CG4Test (OTHER_FAULT)
        159 - cfg4Test.CInterpolationTest (OTHER_FAULT)
        162 - okg4Test.OKG4Test (OTHER_FAULT)
    Errors while running CTest
    opticks-t- : use -V to show output



Standard::

    90% tests passed, 16 tests failed out of 164

    Total Test time (real) = 109.84 sec

    The following tests FAILED:
          4 - SysRapTest.SSysTest (Failed)
         34 - NPYTest.NOpenMeshCfgTest (OTHER_FAULT)
         35 - NPYTest.NOpenMeshFindTest (OTHER_FAULT)
         81 - GGeoTest.GPartsTest (OTHER_FAULT)
         82 - GGeoTest.GPmtTest (OTHER_FAULT)
         90 - GGeoTest.GGeoTestTest (OTHER_FAULT)
         91 - GGeoTest.GMakerTest (SEGFAULT)
         97 - GGeoTest.GPropertyTest (OTHER_FAULT)
        103 - GGeoTest.GSceneTest (OTHER_FAULT)
        104 - AssimpRapTest.AssimpRapTest (SEGFAULT)
        138 - OptiXRapTest.OEventTest (OTHER_FAULT)
        139 - OptiXRapTest.OInterpolationTest (Failed)
        143 - OKOPTest.OpSeederTest (OTHER_FAULT)
        150 - OKTest.VizTest (OTHER_FAULT)
        152 - cfg4Test.CMaterialLibTest (OTHER_FAULT)
        153 - cfg4Test.CTestDetectorTest (OTHER_FAULT)
    Errors while running CTest




Greenfield run errors
---------------------

Were missing a run of `opticks-prepare-installcache`, have added this to `opticks-full`.


::

    2017-06-12 17:43:45.714 INFO  [6775612] [Composition::setCenterExtent@991] Composition::setCenterExtent ce -16520.0000,-802110.0000,-7125.0000,7710.5625
    2017-06-12 17:43:45.715 INFO  [6775612] [SLog::operator@15] OpticksViz::OpticksViz DONE
    2017-06-12 17:43:46.028 INFO  [6775612] [SLog::operator@15] OScene::OScene DONE
    2017-06-12 17:43:46.029 FATAL [6775612] [*OContext::addEntry@44] OContext::addEntry G
    2017-06-12 17:43:46.029 INFO  [6775612] [SLog::operator@15] OEvent::OEvent DONE
    cuRANDWrapper::LoadIntoHostBuffer MISSING RNG CACHE AT : /tmp/blyth/opticks20170612/installcache/RNG/cuRANDWrapper_3000000_0_0.bin 
    cuRANDWrapper::LoadIntoHostBuffer : CREATE CACHE WITH bash functions : cudarap-;cudarap-prepare-installcache 
    cuRANDWrapper::LoadIntoHostBuffer : NB cudarap-prepare-installcache SHOULD HAVE BEEN INVOKED BY opticks-prepare-installcache  
    Assertion failed: (0), function LoadIntoHostBuffer, file /Users/blyth/opticks/cudarap/cuRANDWrapper.cc, line 342.
    Process 68932 stopped
    * thread #1: tid = 0x67633c, 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff8f018866 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill + 10:
    -> 0x7fff8f018866:  jae    0x7fff8f018870            ; __pthread_kill + 20
       0x7fff8f018868:  movq   %rax, %rdi
       0x7fff8f01886b:  jmp    0x7fff8f015175            ; cerror_nocancel
       0x7fff8f018870:  retq   
    (lldb) 





Error Report from Axel
------------------------

On Jun 9, 2017, 

Hi Simon,

this is the main problem i get at the moment:


    axel@axel-VirtualBox ~/opticks $ opticks- ; opticks-full
    === opticks-full : START Wed Jun 7 16:42:38 CEST 2017
    bash: opticks.bash: No such file or directory
    === opticks-cmake : opticks-prefix:/usr/local/opticks
    === opticks-cmake : opticks-optix-install-dir:/tmp
    === opticks-cmake : g4-cmake-dir:/usr/local/opticks/externals/lib/Geant4-10.2.1
    === opticks-cmake : xercesc-library:/usr/local/opticks/externals/lib/libxerces-c-3-1.so
    === opticks-cmake : xercesc-include-dir:/usr/local/opticks/externals/include
    -- The C compiler identification is GNU 5.4.0
    -- The CXX compiler identification is GNU 5.4.0
    -- Check for working C compiler: /usr/bin/cc
    -- Check for working C compiler: /usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /usr/bin/c++
    -- Check for working CXX compiler: /usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring Opticks
    CMAKE_BUILD_TYPE = Debug
    CMAKE_CXX_FLAGS =  -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe
    CMAKE_CXX_FLAGS_DEBUG = -g -DG4FPE_DEBUG
    CMAKE_CXX_FLAGS_RELEASE = -O2 -DNDEBUG
    CMAKE_CXX_FLAGS_RELWITHDEBINFO= -O2 -g
    -- Boost version: 1.58.0
    -- Found the following Boost libraries:
    --   system
    --   program_options
    --   filesystem
    --   regex
    -- Configuring SysRap
    -- Configuring BoostRap
    -- Configuring NPY
    NPY.CSGBSP_INCLUDE_DIRS:/usr/local/opticks/externals/csgbsp/csgjs-cpp
    -- Configuring OpticksCore
    -- Configuring GGeo
    -- Configuring AssimpRap
    -- Configuring OpenMeshRap
    -- Configuring OpticksGeometry
    -- Configuring OGLRap
    -- Opticks.COMPUTE_CAPABILITY : 0 : at least 30 is required for Opticks, proceeding GPU-less
    -- Configuring OK
    Operating without OPTIX
    Opticks.Geant4_FOUND_NOT
    CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
    Please set them or make sure they are set and tested correctly in the CMake files:
    DualContouringSample_LIBRARIES
       linked by target "NPY" in directory /home/axel/opticks/opticksnpy
       linked by target "NPolygonizerTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NOpenMeshFindTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NOpenMeshCfgTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NYShapeTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NYMathTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NTrisTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "HitsNPYTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NBBoxTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NCSGDeserializeTest" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NLoadCheck" in directory /home/axel/opticks/opticksnpy/tests
       linked by target "NTreeToolsTest" in directory /home/axel/opticks/opticksnpy/tests
    ...




Checking Green Field Opticks Installation
--------------------------------------------

* add to ~/.bash_profile a temporary envvar, export OPTICKS_GREENFIELD_TEST=1

This changes the result of opticks-prefix, so can test geenfield building 
into a day stamped folder.


::

    simon:~ blyth$ opticks-prefix
    /usr/local/opticks
    simon:~ blyth$ OPTICKS_GREENFIELD_TEST= opticks-prefix
    /usr/local/opticks
    simon:~ blyth$ OPTICKS_GREENFIELD_TEST=1 opticks-prefix
    /usr/local/opticks20170612



BUT CMake has some hardcoded paths ??
----------------------------------------

Some opticks/cmake/Modules/ were using `$ENV`, for "official" opticks
externals : should not depend on environment in this way...


Tests using optional externals, need detection of the externals presence
-------------------------------------------------------------------------

* some currently optionals need to be moved to standard externals
* many tests need optional inclusion


::

    -- cfg4._line #define G4VERSION_NUMBER  1021 ===> 1021 
    -- Configuring okg4
    CMake Error: The following variables are used in this project, but they are set to NOTFOUND.
    Please set them or make sure they are set and tested correctly in the CMake files:
    ImplicitMesher_LIBRARIES
        linked by target "NPY" in directory /Users/blyth/opticks/opticksnpy
        linked by target "NPolygonizerTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NCSGBSPTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NuvTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshCombineTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshCfgTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshFindTest" in directory /Users/blyth/opticks/opticksnpy/tests
        linked by target "NOpenMeshTest" in directory /Users/blyth/opticks/opticksnpy/tests

    -- Configuring incomplete, errors occurred!
    See also "/tmp/blyth/opticks20170612/build/CMakeFiles/CMakeOutput.log".
    make: *** No rule to make target `install'.  Stop.
    === opticks-full : DONE Mon Jun 12 13:41:56 CST 2017
    simon:~ blyth$ 








