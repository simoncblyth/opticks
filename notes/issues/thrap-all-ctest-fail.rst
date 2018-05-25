With opticks-cmake-overhaul all thrap tests failing cudaGetDevice
==================================================================


All fail for same error
------------------------

::

    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: get_max_shared_memory_per_block :failed to cudaGetDevice: CUDA driver version is insufficient for CUDA runtime version


* https://github.com/kozyilmaz/nheqminer-macos/issues/16


integrated vs proj-by-proj building
--------------------------------------

proj-by-proj 
    runs find_package for direct dependencies which loads the persisted config, each of 
    which in turn runs find_dependency for its direct dependencies and so on 

    find_package for most externals are hooked up via cmake/Modules/FindNAME.cmake

integrated
    find_package is redefined to a no-op (via a macro) for packages that 
    are hooked up via add_subdirectory, externals 


::

    opticks-deps --tree


    NPY
        BCM
        PLog        << needed ? comes with SysRap (integrated?)
        GLM
        OpenMesh
        BoostRap
        YoctoGL
        ImplicitMesher
        DualContouringSample
    OpticksCore
        BCM
        OKConf
        NPY
    ...
    CUDARap
        BCM
        SysRap
        OpticksCUDA      ## this does the find_package(CUDA) and hence defines cuda_add_library cuda_add_executable 
    ThrustRap
        BCM
        OpticksCore      ## find_package(OpticksCore CONFIG REQUIRED )
        CUDARap          ## find_package(CUDARap CONFIG REQUIRED )       
                         ##     both these are skipped for integrated build (by selective noop of find_package macro), 
                         ##     thus config for these relies on having just built the targets via add_subdirectory 


* the integrated build of ThrustRap skips the find


zoom in on a simple test expandTest 
----------------------------------------------

The one that works::

    epsilon:tests blyth$ pwd
    /usr/local/opticks-cmake-overhaul/build/thrustrap/tests
    epsilon:tests blyth$ touch ~/opticks-cmake-overhaul/thrustrap/tests/expandTest.cu 
    epsilon:tests blyth$ make expandTest


integrated build with flags brought over in EnvCompilationFlags.cmake
-----------------------------------------------------------------------

::

    epsilon:opticks-cmake-overhaul blyth$ ./go.sh 


    opticks-home   : /Users/blyth/opticks-cmake-overhaul
    opticks-prefix : /usr/local/opticks-cmake-overhaul
    opticks-prefix-tmp : /usr/local/opticks-cmake-overhaul-tmp
    opticks-name   : opticks-cmake-overhaul


    === ./go.sh : integrated build : for now into opticks-prefix-tmp : /usr/local/opticks-cmake-overhaul-tmp

    found externals dir or link inside the prefix
    found opticksdata dir inside the prefix
    /usr/local/opticks-cmake-overhaul-tmp
    total 0
    drwxr-xr-x   29 blyth  staff    928 May 24 19:34 build
    lrwxr-xr-x    1 blyth  staff     20 May 24 16:05 externals -> ../opticks/externals
    drwxr-xr-x    3 blyth  staff     96 May 24 17:37 geocache
    drwxr-xr-x   20 blyth  staff    640 May 24 17:38 gl
    drwxr-xr-x   20 blyth  staff    640 May 24 17:43 include
    drwxr-xr-x    3 blyth  staff     96 May 24 16:21 installcache
    drwxr-xr-x  328 blyth  staff  10496 May 24 19:33 lib
    lrwxr-xr-x    1 blyth  staff     22 May 24 16:05 opticksdata -> ../opticks/opticksdata
    /usr/local/opticks-cmake-overhaul-tmp/build
    === ./go.sh : sdir : /Users/blyth/opticks-cmake-overhaul
    === ./go.sh : bdir : /usr/local/opticks-cmake-overhaul-tmp/build
    === ./go.sh : pwd : /usr/local/opticks-cmake-overhaul-tmp/build
    ...

    -- Installing: /usr/local/opticks-cmake-overhaul-tmp/lib/cmake/okg4/okg4-targets-debug.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul-tmp/lib/cmake/okg4/okg4-config.cmake
    -- Up-to-date: /usr/local/opticks-cmake-overhaul-tmp/lib/cmake/okg4/okg4-config-version.cmake
    -- Installing: /usr/local/opticks-cmake-overhaul-tmp/lib/OKG4Test
    === ./go.sh : make installing 32 seconds ie 0 minutes
    === ./go.sh :              cmake configuring 5 seconds ie 0 minutes 
    === ./go.sh :                  make building 851 seconds ie 14 minutes 
    === ./go.sh :                make installing 32 seconds ie 0 minutes 
    epsilon:opticks-cmake-overhaul blyth$ 

    epsilon:opticks-cmake-overhaul blyth$ opticks-t /usr/local/opticks-cmake-overhaul-tmp/build

    ...

    92% tests passed, 25 tests failed out of 299

    Total Test time (real) =  92.99 sec

    The following tests FAILED:
        190 - GGeoTest.GBndLibInitTest (SEGFAULT)
        222 - GGeoTest.GSceneTest (Child aborted)
        235 - ThrustRapTest.CBufSpecTest (Child aborted)
        236 - ThrustRapTest.TBufTest (Child aborted)
        237 - ThrustRapTest.TRngBufTest (Child aborted)
        238 - ThrustRapTest.expandTest (Child aborted)
        239 - ThrustRapTest.iexpandTest (Child aborted)
        240 - ThrustRapTest.issue628Test (Child aborted)
        241 - ThrustRapTest.printfTest (Child aborted)
        242 - ThrustRapTest.repeated_rangeTest (Child aborted)
        243 - ThrustRapTest.strided_rangeTest (Child aborted)
        244 - ThrustRapTest.strided_repeated_rangeTest (Child aborted)
        245 - ThrustRapTest.float2intTest (Child aborted)
        246 - ThrustRapTest.thrust_curand_estimate_pi (Child aborted)
        247 - ThrustRapTest.thrust_curand_printf (Child aborted)
        248 - ThrustRapTest.thrust_curand_printf_redirect (Child aborted)
        249 - ThrustRapTest.thrust_curand_printf_redirect2 (Child aborted)
        265 - OptiXRapTest.ORayleighTest (Child aborted)
        269 - OKOPTest.OpSeederTest (Child aborted)
        276 - OKTest.OKTest (Child aborted)
        282 - CFG4Test.CTestDetectorTest (Child aborted)
        285 - CFG4Test.CG4Test (Child aborted)
        293 - CFG4Test.CInterpolationTest (Child aborted)
        298 - CFG4Test.CRandomEngineTest (Child aborted)
        299 - OKG4Test.OKG4Test (Child aborted)
    Errors while running CTest
    Thu May 24 20:27:26 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul-tmp/build/ctest.log
    epsilon:opticks-cmake-overhaul blyth$ 




integrated vs subproj thrustrap flag check
---------------------------------------------

* 


integrated::

    epsilon:opticks-cmake-overhaul blyth$ touch thrustrap/THRAP_API_EXPORT.hh
    epsilon:opticks-cmake-overhaul blyth$ export VERBOSE=1
    epsilon:opticks-cmake-overhaul blyth$ ./go.sh 

::

    -- Generating /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TUtil_.cu.o
       /Developer/NVIDIA/CUDA-9.1/bin/nvcc 
      /Users/blyth/opticks-cmake-overhaul/thrustrap/TUtil_.cu 
      -c 
      -o /usr/local/opticks-cmake-overhaul-tmp/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TUtil_.cu.o 
        -ccbin /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang 
        -m64 
       -DThrustRap_EXPORTS 
       -DWITH_YoctoGL 
       -DWITH_ImplicitMesher 
       -DWITH_DualContouringSample 
       -Xcompiler 
           ,\"-fvisibility=hidden\"
           ,\"-Wall\"
           ,\"-Wno-unused-function\"
           ,\"-Wno-unused-private-field\"
           ,\"-Wno-shadow\"
           ,\"-fPIC\" 
      -DNVCC 
         -I/Users/blyth/opticks-cmake-overhaul/thrustrap 
         -I/Users/blyth/opticks-cmake-overhaul/optickscore 
         -I/Users/blyth/opticks-cmake-overhaul/npy 
         -I/usr/local/opticks-cmake-overhaul-tmp/externals/glm/glm 
         -I/Users/blyth/opticks-cmake-overhaul/sysrap 
         -I/Users/blyth/opticks-cmake-overhaul/sysrap/include  ## this was a stray INCLUDE inside bcm_deploy 
         -I/usr/local/opticks-cmake-overhaul-tmp/externals/plog/include 
         -I/usr/local/opticks-cmake-overhaul-tmp/build/boostrap/inc    ## some more stray INCLUDE inside bcm_deploy 
         -I/Users/blyth/opticks-cmake-overhaul/boostrap 
         -I/usr/local/opticks-cmake-overhaul-tmp/build/boostrap 

         -I/opt/local/include 
         -I/usr/local/opticks-cmake-overhaul/externals/include 
         -I/usr/local/opticks-cmake-overhaul/externals/include/YoctoGL 
         -I/usr/local/opticks-cmake-overhaul/externals/include/DualContouringSample 
               ## external includes are same 

         -I/usr/local/opticks-cmake-overhaul-tmp/build/okconf/inc 
         -I/Users/blyth/opticks-cmake-overhaul/okconf  
         -I/Users/blyth/opticks-cmake-overhaul/cudarap 
         -I/Developer/NVIDIA/CUDA-9.1/include

               ## hmm cuda comes last for integrated ??? but first for proj-by-proj

       ^Cmake[2]: *** [thrustrap/CMakeFiles/ThrustRap.dir/ThrustRap_generated_TUtil_.cu.o] Interrupt: 2


proj-by-proj (so targets are imported)::

   -- Generating /usr/local/opticks-cmake-overhaul/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TUtil_.cu.o
    /Developer/NVIDIA/CUDA-9.1/bin/nvcc 
      /Users/blyth/opticks-cmake-overhaul/thrustrap/TUtil_.cu 
     -c 
    -o /usr/local/opticks-cmake-overhaul/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TUtil_.cu.o 
     -ccbin /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang 
      -m64 
     -DThrustRap_EXPORTS 
     -DWITH_YoctoGL -DWITH_ImplicitMesher -DWITH_DualContouringSample
       -Xcompiler 
         ,\"-fvisibility=hidden\",\"-Wall\",\"-Wno-unused-function\",\"-Wno-unused-private-field\",\"-Wno-shadow\",\"-fPIC\"
         ,\"-g\"
        -DNVCC
           -I/Developer/NVIDIA/CUDA-9.1/include 
           -I/Users/blyth/opticks-cmake-overhaul/thrustrap 
           -I/usr/local/opticks-cmake-overhaul/include/OpticksCore 
           -I/usr/local/opticks-cmake-overhaul/include/NPY 
           -I/usr/local/opticks-cmake-overhaul/externals/glm/glm 
           -I/usr/local/opticks-cmake-overhaul/include/SysRap 
           -I/usr/local/opticks-cmake-overhaul/externals/plog/include 
           -I/usr/local/opticks-cmake-overhaul/include/BoostRap 
           -I/opt/local/include 

           -I/usr/local/opticks-cmake-overhaul/externals/include 
           -I/usr/local/opticks-cmake-overhaul/externals/include/YoctoGL 
           -I/usr/local/opticks-cmake-overhaul/externals/include/DualContouringSample 

           -I/usr/local/opticks-cmake-overhaul/include/OKConf 
           -I/usr/local/opticks-cmake-overhaul/include/CUDARap

     Generated /usr/local/opticks-cmake-overhaul/build/thrustrap/CMakeFiles/ThrustRap.dir//./ThrustRap_generated_TUtil_.cu.o successfully.



::

    epsilon:thrustrap blyth$ vex
    vex: export VERBOSE=1
    epsilon:thrustrap blyth$ touch TUtil_.cu
    epsilon:thrustrap blyth$ ./go.sh 




proj-by-proj wrong number of tests
--------------------------------------


* when building proj-by-proj you should really be running 
  ctest proj-by-proj ... it only managed to do something at top level
  due to a prior old integrated build presumably from May 15 



::

    epsilon:build blyth$ pwd
    /usr/local/opticks-cmake-overhaul/build

    epsilon:build blyth$ l
    total 592
    drwxr-xr-x  15 blyth  staff  -    480 May 24 18:50 npy
    -rw-r--r--   1 blyth  staff  -  26923 May 24 18:46 ctest.log
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 okg4
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 cfg4
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 ok
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 opticksgl
    drwxr-xr-x  15 blyth  staff  -    480 May 24 18:41 oglrap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 okop
    drwxr-xr-x  42 blyth  staff  -   1344 May 24 18:41 optixrap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 thrustrap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 cudarap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 opticksgeo
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 openmeshrap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:41 assimprap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:40 ggeo
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:40 optickscore
    drwxr-xr-x  15 blyth  staff  -    480 May 24 18:40 boostrap
    drwxr-xr-x  14 blyth  staff  -    448 May 24 18:40 sysrap
    drwxr-xr-x  15 blyth  staff  -    480 May 24 18:40 okconf
    -rw-r--r--   1 blyth  staff  -  73382 May 23 22:44 CMakeCache.txt
    drwxr-xr-x  46 blyth  staff  -   1472 May 23 22:43 CMakeFiles
    drwxr-xr-x   3 blyth  staff  -     96 May 17 22:03 inc
    drwxr-xr-x   3 blyth  staff  -     96 May 17 22:03 include
    -rw-r--r--   1 blyth  staff  - 116789 May 17 20:49 Makefile
    -rw-r--r--   1 blyth  staff  -  40989 May 15 19:13 install_manifest.txt
    -rw-r--r--   1 blyth  staff  -    613 May 15 16:37 CTestTestfile.cmake
    -rw-r--r--   1 blyth  staff  -   4480 May 15 16:37 cmake_install.cmake
    -rwxr-xr-x   1 blyth  staff  -    547 May 15 16:37 OpticksConfig.cmake
    -rwxr-xr-x   1 blyth  staff  -   4961 May 15 16:37 opticks-config
    drwxr-xr-x   4 blyth  staff  -    128 May 15 16:37 test
    -rw-r--r--   1 blyth  staff  -   4017 May 15 16:37 CPackSourceConfig.cmake
    -rw-r--r--   1 blyth  staff  -   3560 May 15 16:37 CPackConfig.cmake
    -rw-r--r--   1 blyth  staff  -   2850 May 15 16:37 DartConfiguration.tcl
    drwxr-xr-x   3 blyth  staff  -     96 May 15 16:37 Testing
    epsilon:build blyth$ vi CTestTestfile.cmake 




proj-by-proj again : 6/196 (101 npy are skipped?)
-----------------------------------------------------

::

    epsilon:build blyth$ pwd
    /usr/local/opticks-cmake-overhaul/build
    epsilon:build blyth$ opticks-
    epsilon:build blyth$ opticks-t $PWD

::

    97% tests passed, 6 tests failed out of 196

    Total Test time (real) = 123.09 sec

    The following tests FAILED:
        119 - GGeoTest.GSceneTest (Child aborted)
        179 - CFG4Test.CTestDetectorTest (Child aborted)
        182 - CFG4Test.CG4Test (Child aborted)
        190 - CFG4Test.CInterpolationTest (Child aborted)
        195 - CFG4Test.CRandomEngineTest (Child aborted)
        196 - OKG4Test.OKG4Test (Child aborted)
    Errors while running CTest
    Thu May 24 18:46:23 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul/build/ctest.log
    epsilon:build blyth$ 




gosub.sh proj-by-proj building
---------------------------------


::

     o  # cd to opticks-home ~/opticks-cmake-overhaul

     ./gosub.sh  # proj by proj building 

     cd /usr/local/opticks-cmake-overhaul/build

     opticks-executables | wc -l   ## counting executables yields expected 300 (now 304)

     ctest   ## ctest running only runs 192 ??? ahha trivial comment out of NPY tests



::

    97% tests passed, 6 tests failed out of 192

    Total Test time (real) = 278.99 sec

    The following tests FAILED:
        119 - GGeoTest.GSceneTest (Child aborted)
        175 - CFG4Test.CTestDetectorTest (Child aborted)
        178 - CFG4Test.CG4Test (Child aborted)
        186 - CFG4Test.CInterpolationTest (Child aborted)
        191 - CFG4Test.CRandomEngineTest (Child aborted)
        192 - OKG4Test.OKG4Test (Child aborted)
    Errors while running CTest



integrated build testing : 24/299 failed
--------------------------------------------

::

    ./go.sh ## with prefix=$(opticks-prefix-tmp)   /usr/local/opticks-cmake-overhaul-tmp


    92% tests passed, 24 tests failed out of 299

    Total Test time (real) =  93.45 sec

    The following tests FAILED:
        222 - GGeoTest.GSceneTest (Child aborted)
        235 - ThrustRapTest.CBufSpecTest (Child aborted)
        236 - ThrustRapTest.TBufTest (Child aborted)
        237 - ThrustRapTest.TRngBufTest (Child aborted)
        238 - ThrustRapTest.expandTest (Child aborted)
        239 - ThrustRapTest.iexpandTest (Child aborted)
        240 - ThrustRapTest.issue628Test (Child aborted)
        241 - ThrustRapTest.printfTest (Child aborted)
        242 - ThrustRapTest.repeated_rangeTest (Child aborted)
        243 - ThrustRapTest.strided_rangeTest (Child aborted)
        244 - ThrustRapTest.strided_repeated_rangeTest (Child aborted)
        245 - ThrustRapTest.float2intTest (Child aborted)
        246 - ThrustRapTest.thrust_curand_estimate_pi (Child aborted)
        247 - ThrustRapTest.thrust_curand_printf (Child aborted)
        248 - ThrustRapTest.thrust_curand_printf_redirect (Child aborted)
        249 - ThrustRapTest.thrust_curand_printf_redirect2 (Child aborted)
        265 - OptiXRapTest.ORayleighTest (Child aborted)
        269 - OKOPTest.OpSeederTest (Child aborted)
        276 - OKTest.OKTest (Child aborted)
        282 - CFG4Test.CTestDetectorTest (Child aborted)
        285 - CFG4Test.CG4Test (Child aborted)
        293 - CFG4Test.CInterpolationTest (Child aborted)
        298 - CFG4Test.CRandomEngineTest (Child aborted)
        299 - OKG4Test.OKG4Test (Child aborted)
    Errors while running CTest
    Thu May 24 17:45:58 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul-tmp/build/ctest.log
    epsilon:build blyth$ 
    epsilon:build blyth$ 
    epsilon:build blyth$ pwd
    /usr/local/opticks-cmake-overhaul-tmp/build
    epsilon:build blyth$ opticks-t $PWD




Difference between the built and installed thrap binaries ?
--------------------------------------------------------------

* ctest : runs the build dir binaries 



Switching Between Opticks versions
-------------------------------------

Switch between opticks versions by changing OPTICKS_HOME in .bash_profile and starting new bash tab::

    319 export OPTICKS_HOME=$HOME/opticks
    320 #export OPTICKS_HOME=$HOME/opticks-cmake-overhaul
    321 
    322 opticks-(){  [ -r $OPTICKS_HOME/opticks.bash ] && . $OPTICKS_HOME/opticks.bash && opticks-env $* && opticks-export ; }
    323 


Old Opticks::

    epsilon:issues blyth$ which CBufSpecTest
    /usr/local/opticks/lib/CBufSpecTest

New Opticks::

    epsilon:issues blyth$ which CBufSpecTest
    /usr/local/opticks-cmake-overhaul/lib/CBufSpecTest

    * also possible there is difference between integrated and subproj CMake builds ?

    * somehow the integrated and proj builds somehow getting different nvcc flags ?






Huh building subproj with new Opticks doesnt have the issue ?
-----------------------------------------------------------------


::

    thrap-cd
    ./go.sh

    epsilon:thrustrap blyth$ thrap-t 
    Thu May 24 14:13:12 HKT 2018
    Test project /usr/local/opticks-cmake-overhaul/build/thrustrap
          Start  1: ThrustRapTest.CBufSpecTest
     1/15 Test  #1: ThrustRapTest.CBufSpecTest .....................   Passed    0.98 sec
          Start  2: ThrustRapTest.TBufTest
     2/15 Test  #2: ThrustRapTest.TBufTest .........................   Passed    1.00 sec
          Start  3: ThrustRapTest.TRngBufTest
     3/15 Test  #3: ThrustRapTest.TRngBufTest ......................   Passed    2.16 sec
          Start  4: ThrustRapTest.expandTest
     4/15 Test  #4: ThrustRapTest.expandTest .......................   Passed    1.04 sec
          Start  5: ThrustRapTest.iexpandTest
     5/15 Test  #5: ThrustRapTest.iexpandTest ......................   Passed    0.95 sec
          Start  6: ThrustRapTest.issue628Test
     6/15 Test  #6: ThrustRapTest.issue628Test .....................   Passed    0.92 sec
          Start  7: ThrustRapTest.printfTest
     7/15 Test  #7: ThrustRapTest.printfTest .......................   Passed    0.96 sec
          Start  8: ThrustRapTest.repeated_rangeTest
     8/15 Test  #8: ThrustRapTest.repeated_rangeTest ...............   Passed    1.17 sec
          Start  9: ThrustRapTest.strided_rangeTest
     9/15 Test  #9: ThrustRapTest.strided_rangeTest ................   Passed    0.98 sec
          Start 10: ThrustRapTest.strided_repeated_rangeTest
    10/15 Test #10: ThrustRapTest.strided_repeated_rangeTest .......   Passed    1.22 sec
          Start 11: ThrustRapTest.float2intTest
    11/15 Test #11: ThrustRapTest.float2intTest ....................   Passed    1.22 sec
          Start 12: ThrustRapTest.thrust_curand_estimate_pi
    12/15 Test #12: ThrustRapTest.thrust_curand_estimate_pi ........   Passed    1.32 sec
          Start 13: ThrustRapTest.thrust_curand_printf
    13/15 Test #13: ThrustRapTest.thrust_curand_printf .............   Passed    0.92 sec
          Start 14: ThrustRapTest.thrust_curand_printf_redirect
    14/15 Test #14: ThrustRapTest.thrust_curand_printf_redirect ....   Passed    1.13 sec
          Start 15: ThrustRapTest.thrust_curand_printf_redirect2
    15/15 Test #15: ThrustRapTest.thrust_curand_printf_redirect2 ...   Passed    0.95 sec

    100% tests passed, 0 tests failed out of 15

    Total Test time (real) =  16.91 sec
    Thu May 24 14:13:29 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks-cmake-overhaul/build/thrustrap/ctest.log
    epsilon:thrustrap blyth$ which CBufSpecTest
    /usr/local/opticks-cmake-overhaul/lib/CBufSpecTest
    epsilon:thrustrap blyth$ 



Old opticks
----------------

With the old Opticks::

    epsilon:opticks blyth$ thrap-t
    Thu May 24 13:59:22 HKT 2018
    Test project /usr/local/opticks/build/thrustrap
          Start  1: ThrustRapTest.CBufSpecTest
     1/15 Test  #1: ThrustRapTest.CBufSpecTest .....................   Passed    0.85 sec
          Start  2: ThrustRapTest.TBufTest
     2/15 Test  #2: ThrustRapTest.TBufTest .........................   Passed    0.93 sec
          Start  3: ThrustRapTest.TRngBufTest
     3/15 Test  #3: ThrustRapTest.TRngBufTest ......................   Passed    1.73 sec
          Start  4: ThrustRapTest.expandTest
     4/15 Test  #4: ThrustRapTest.expandTest .......................   Passed    0.87 sec
          Start  5: ThrustRapTest.iexpandTest
     5/15 Test  #5: ThrustRapTest.iexpandTest ......................   Passed    0.92 sec
          Start  6: ThrustRapTest.issue628Test
     6/15 Test  #6: ThrustRapTest.issue628Test .....................   Passed    0.94 sec
          Start  7: ThrustRapTest.printfTest
     7/15 Test  #7: ThrustRapTest.printfTest .......................   Passed    1.07 sec
          Start  8: ThrustRapTest.repeated_rangeTest
     8/15 Test  #8: ThrustRapTest.repeated_rangeTest ...............   Passed    1.00 sec
          Start  9: ThrustRapTest.strided_rangeTest
     9/15 Test  #9: ThrustRapTest.strided_rangeTest ................   Passed    0.95 sec
          Start 10: ThrustRapTest.strided_repeated_rangeTest
    10/15 Test #10: ThrustRapTest.strided_repeated_rangeTest .......   Passed    1.05 sec
          Start 11: ThrustRapTest.float2intTest
    11/15 Test #11: ThrustRapTest.float2intTest ....................   Passed    1.02 sec
          Start 12: ThrustRapTest.thrust_curand_estimate_pi
    12/15 Test #12: ThrustRapTest.thrust_curand_estimate_pi ........   Passed    1.37 sec
          Start 13: ThrustRapTest.thrust_curand_printf
    13/15 Test #13: ThrustRapTest.thrust_curand_printf .............   Passed    0.95 sec
          Start 14: ThrustRapTest.thrust_curand_printf_redirect
    14/15 Test #14: ThrustRapTest.thrust_curand_printf_redirect ....   Passed    1.13 sec
          Start 15: ThrustRapTest.thrust_curand_printf_redirect2
    15/15 Test #15: ThrustRapTest.thrust_curand_printf_redirect2 ...   Passed    1.16 sec

    100% tests passed, 0 tests failed out of 15

    Total Test time (real) =  15.96 sec
    Thu May 24 13:59:38 HKT 2018
    === opticks-t- : use -V to show output, ctest output written to /usr/local/opticks/build/thrustrap/ctest.log
    epsilon:opticks blyth$ 


New Opticks
-------------

::

    cd /tmp/build/thrap
    ctest

    0% tests passed, 15 tests failed out of 15

    Total Test time (real) =   0.42 sec

    The following tests FAILED:
          1 - ThrustRapTest.CBufSpecTest (Child aborted)
          2 - ThrustRapTest.TBufTest (Child aborted)
          3 - ThrustRapTest.TRngBufTest (Child aborted)
          4 - ThrustRapTest.expandTest (Child aborted)
          5 - ThrustRapTest.iexpandTest (Child aborted)
          6 - ThrustRapTest.issue628Test (Child aborted)
          7 - ThrustRapTest.printfTest (Child aborted)
          8 - ThrustRapTest.repeated_rangeTest (Child aborted)
          9 - ThrustRapTest.strided_rangeTest (Child aborted)
         10 - ThrustRapTest.strided_repeated_rangeTest (Child aborted)
         11 - ThrustRapTest.float2intTest (Child aborted)
         12 - ThrustRapTest.thrust_curand_estimate_pi (Child aborted)
         13 - ThrustRapTest.thrust_curand_printf (Child aborted)
         14 - ThrustRapTest.thrust_curand_printf_redirect (Child aborted)
         15 - ThrustRapTest.thrust_curand_printf_redirect2 (Child aborted)
    Errors while running CTest
    epsilon:thrustrap blyth$ CBufSpecTest
    libc++abi.dylib: terminating with uncaught exception of type thrust::system::system_error: get_max_shared_memory_per_block :failed to cudaGetDevice: CUDA driver version is insufficient for CUDA runtime version
    Abort trap: 6




