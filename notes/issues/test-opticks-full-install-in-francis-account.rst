test-opticks-full-install-in-francis-account
=============================================


Attempt 1 : "git clone" then "opticks-full" : get network fail fail for glew tarball
----------------------------------------------------------------------------------------

::

    epsilon:~ francis$ mv opticks blyth_opticks
    
    epsilon:~ francis$ cd /Users/francis/local/
    epsilon:local francis$ l
    total 0
    drwxr-xr-x  12 francis  staff  384 Jan 17  2021 opticks
    lrwxr-xr-x   1 francis  staff   28 Jan 17  2021 opticks_externals -> /usr/local/opticks_externals
    epsilon:local francis$ mv opticks blyth_opticks
    epsilon:local francis$ mkdir opticks
    epsilon:local francis$ 



    epsilon:~ francis$ opticks-
    epsilon:~ francis$ opticks-full
    opticks-externals-info
    ============================

        opticks-cmake-version  : 3.17.1


    opticks-locations
    ==================

          OPTICKS_PREFIX  :    /Users/francis/local/opticks
          opticks-prefix  :    /Users/francis/local/opticks
          #opticks-optix-install-dir :  
          opticks-optix-prefix :  /usr/local/optix
          opticks-cuda-prefix :  /usr/local/cuda



          opticks-source   :   /Users/francis/opticks/opticks.bash
          opticks-home     :   /Users/francis/opticks
          opticks-name     :   opticks


          opticks-sdir     :   /Users/francis/opticks
          opticks-idir     :   /Users/francis/local/opticks
          opticks-bdir     :   /Users/francis/local/opticks/build
          opticks-xdir     :   /Users/francis/local/opticks/externals
          ## cd to these with opticks-scd/icd/bcd/xcd

          opticks-installcachedir   :  /Users/francis/local/opticks/installcache
          opticks-bindir            :  /Users/francis/local/opticks/lib


           uname   : Darwin epsilon.local 17.7.0 Darwin Kernel Version 17.7.0: Thu Jun 21 22:53:14 PDT 2018; root:xnu-4570.71.2~1/RELEASE_X86_64 x86_64
           HOME    : /Users/francis
           VERBOSE : 
           USER    : francis

    OPTICKS_TMP=/tmp/francis/opticks
    OPTICKS_PREFIX=/Users/francis/local/opticks
    OPTICKS_EVENT_BASE=/tmp/francis/opticks
    OPTICKS_COMPUTE_CAPABILITY=30
    OPTICKS_GEANT4_PREFIX=/usr/local/opticks_externals/g4_1042
    OPTICKS_HOME=/Users/francis/opticks
    OPTICKS_OPTIX_PREFIX=/usr/local/optix
    OPTICKS_CUDA_PREFIX=/usr/local/cuda
    opticks-externals-url
                               bcm :  http://github.com/simoncblyth/bcm.git 
                               glm :  https://github.com/g-truc/glm/releases/download/0.9.9.5/glm-0.9.9.5.zip 
                              glfw :  https://github.com/glfw/glfw/releases/download/3.3.2/glfw-3.3.2.zip 
                              glew :  http://downloads.sourceforge.net/project/glew/glew/1.13.0/glew-1.13.0.zip 
                              gleq :  https://github.com/simoncblyth/gleq.git 
                             imgui :  http://github.com/simoncblyth/imgui.git 
                              plog :  https://github.com/simoncblyth/plog.git 
                        opticksaux :  https://bitbucket.org/simoncblyth/opticksaux.git 
                            nljson :  https://github.com/nlohmann/json/releases/download/v3.9.1/json.hpp 
    opticks-externals-dist
                               bcm :   
                               glm :  /Users/francis/local/opticks/externals/glm/glm-0.9.9.5.zip 
                              glfw :  /Users/francis/local/opticks/externals/glfw/glfw-3.3.2.zip 
                              glew :  /Users/francis/local/opticks/externals/glew/glew-1.13.0.zip 
                              gleq :   
                             imgui :   
                              plog :   
                        opticksaux :   
                            nljson :  /Users/francis/local/opticks/externals/include/nljson/json.hpp 
    opticks-foreign-url
                             boost :  http://downloads.sourceforge.net/project/boost/boost/1.70.0/boost_1_70_0.tar.gz 
                             clhep :  https://proj-clhep.web.cern.ch/proj-clhep/dist1/clhep-2.4.5.1.tgz 
                           xercesc :  http://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.1.tar.gz 
                                g4 :  http://geant4-data.web.cern.ch/geant4-data/releases/geant4.10.04.p02.tar.gz 
    opticks-foreign-dist
                             boost :   
                             clhep :   
                           xercesc :  xerces-c-3.1.1.tar.gz 
                                g4 :  /usr/local/opticks_externals/g4_1042.build/geant4.10.04.p02.tar.gz 
    === opticks-full-externals : START Fri Oct 20 09:52:39 PST 2023
    === opticks-full-externals : installing the below externals into /Users/francis/local/opticks/externals


    ...

    ############## glew ###############


    === opticks-curl : dir /Users/francis/local/opticks/externals/glew url http://downloads.sourceforge.net/project/glew/glew/1.13.0/glew-1.13.0.zip dist glew-1.13.0.zip OPTICKS_DOWNLOAD_CACHE cmd curl -L -O http://downloads.sourceforge.net/project/glew/glew/1.13.0/glew-1.13.0.zip
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   335  100   335    0     0    799      0 --:--:-- --:--:-- --:--:--   797
    100   154  100   154    0     0    128      0  0:00:01  0:00:01 --:--:--     0
    100   343  100   343    0     0    242      0  0:00:01  0:00:01 --:--:--   242
      0     0    0     0    0     0      0      0 --:--:--  0:01:16 --:--:--     0
    curl: (28) Failed to connect to altushost-swe.dl.sourceforge.net port 80: Operation timed out
    unzip:  cannot find or open glew-1.13.0.zip, glew-1.13.0.zip.zip or glew-1.13.0.zip.ZIP.
    === glew-- : get FAIL
    === opticks-installer- : RC 1 from pkg glew : ABORTING
    === opticks-full : ERR from opticks-full-externals
    epsilon:glew francis$ 




Attempt 2 : keep old externals but delete the rest of the build then run opticks-full
---------------------------------------------------------------------------------------


Use former install of externals, but delete the rest of the build::

    epsilon:opticks francis$ pwd
    /Users/francis/local/opticks

    epsilon:opticks francis$ l
    total 0
    drwxr-xr-x   34 francis  staff   1088 Jul 12  2021 bin
    drwxr-xr-x   24 francis  staff    768 Jan 17  2021 build
    drwxr-xr-x   12 francis  staff    384 Jan 17  2021 externals
    drwxr-xr-x   19 francis  staff    608 Jan 17  2021 gl
    drwxr-xr-x   20 francis  staff    640 Jan 17  2021 include
    drwxr-xr-x    3 francis  staff     96 Jan 17  2021 installcache
    drwxr-xr-x    4 francis  staff    128 Jan 17  2021 integration
    drwxr-xr-x  509 francis  staff  16288 Jul 12  2021 lib
    drwxr-xr-x    6 francis  staff    192 Jan 17  2021 opticksaux
    drwxr-xr-x    3 francis  staff     96 Jan 17  2021 py
    epsilon:opticks francis$ 

    epsilon:opticks francis$ rm -rf bin build gl include installcache integration lib opticksaux py 
    epsilon:opticks francis$ l 
    total 0
    drwxr-xr-x  12 francis  staff  384 Jan 17  2021 externals
    epsilon:opticks francis$ 




Succeed to reproduce the issue::

    -- Found CUDA: /Developer/NVIDIA/CUDA-9.1 (found version "9.1") 
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- CMAKE_INSTALL_PREFIX     : /Users/francis/local/opticks
    -- CMAKE_INSTALL_INCLUDEDIR : include/CSG
    -- CMAKE_INSTALL_LIBDIR     : lib
    -- CMAKE_BINARY_DIR         : /Users/francis/local/opticks/build/CSG
    -- bcm_auto_pkgconfig_each LIB:Threads::Threads : MISSING LIB_PKGCONFIG_NAME 
    CMake Error at tests/CMakeLists.txt:5 (find_package):
      Could not find a package configuration file provided by "OpticksCore" with
      any of the following names:

        OpticksCoreConfig.cmake
        optickscore-config.cmake

      Add the installation prefix of "OpticksCore" to CMAKE_PREFIX_PATH or set
      "OpticksCore_DIR" to a directory containing one of the above files.  If
      "OpticksCore" provides a separate development package or SDK, be sure it
      has been installed.


    -- Configuring incomplete, errors occurred!
    See also "/Users/francis/local/opticks/build/CSG/CMakeFiles/CMakeOutput.log".
    === om-make-one : CSG             /Users/francis/opticks/CSG                                   /Users/francis/local/opticks/build/CSG                       
    === om-make-one : ERROR bdir /Users/francis/local/opticks/build/CSG exists but does not contain a Makefile : you need to om-install OR om-conf once before using om-make or the om-- shortcut
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /Users/francis/local/opticks/build/CSG : non-zero rc 2
    === om-one-or-all install : non-zero rc 2
    === opticks-full : ERR from opticks-full-make
    epsilon:opticks francis$ 


Looks like CSG/tests package has a stale dependency on OpticksCore::

     01 cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
      2 set(name CSGTest)
      3 project(${name} VERSION 0.1.0)
      4 
      5 find_package(OpticksCore REQUIRED CONFIG)
      6 
      7 set(TEST_SOURCES
      8     CSGNodeTest.cc
      9     CSGNodeImpTest.cc
     10     CSGIntersectSolidTest.cc
     11     CSGPrimImpTest.cc



Yep, stray use of the old and now inactive OpticksCore/Opticks.hh::

    epsilon:opticks blyth$ git status
    On branch master
    Your branch is up-to-date with 'origin/master'.

    Changes not staged for commit:
      (use "git add <file>..." to update what will be committed)
      (use "git checkout -- <file>..." to discard changes in working directory)

        modified:   CSG/tests/CMakeLists.txt
        modified:   CSG/tests/CSGFoundry_findSolidIdx_Test.cc
        modified:   CSG/tests/CSGFoundry_getCenterExtent_Test.cc
        modified:   CSG/tests/CSGIntersectSolidTest.cc
        modified:   CSG/tests/CSGTargetGlobalTest.cc
        modified:   opticks.bash

    Untracked files:
      (use "git add <file>..." to include in what will be committed)

        notes/issues/test-opticks-full-install-in-francis-account.rst

    no changes added to commit (use "git add" and/or "git commit -a")
    epsilon:opticks blyth$ 


Fixing up those gets CSG to build. 

Delete the the inactive installed headers to see if any similar
issues are lurking::

    epsilon:lib blyth$ cd ../include
    epsilon:include blyth$ l
    total 0
    0 drwxr-xr-x   14 blyth  staff   448 Oct 16 19:20 PMTSim
    0 drwxr-xr-x   69 blyth  staff  2208 Oct 16 10:15 U4
    0 drwxr-xr-x   40 blyth  staff  1280 Oct 16 10:14 QUDARap
    0 drwxr-xr-x    7 blyth  staff   224 Oct 16 10:13 OKConf
    0 drwxr-xr-x  241 blyth  staff  7712 Oct 16 09:58 SysRap
    0 drwxr-xr-x    5 blyth  staff   160 Sep  4 14:47 G4CX
    0 drwxr-xr-x   65 blyth  staff  2080 Sep  1 21:02 ExtG4
    0 drwxr-xr-x   61 blyth  staff  1952 Sep  1 20:35 GGeo
    0 drwxr-xr-x   44 blyth  staff  1408 Aug 29 00:19 CSG
    0 drwxr-xr-x  139 blyth  staff  4448 Aug 21 23:46 NPY
    0 drwxr-xr-x    5 blyth  staff   160 Jul 30 02:15 CSGOptiX
    0 drwxr-xr-x   32 blyth  staff  1024 Jul 22 02:36 .
    0 drwxr-xr-x    5 blyth  staff   160 Jul 15 17:51 CSG_GGeo
    0 drwxr-xr-x   23 blyth  staff   736 Feb 25  2023 PMTFastSim
    0 drwxr-xr-x    6 blyth  staff   192 Jan 22  2023 CSG_U4
    0 drwxr-xr-x   65 blyth  staff  2080 Jan 16  2023 OpticksCore
    0 drwxr-xr-x   42 blyth  staff  1344 Nov 13  2022 BoostRap
    0 drwxr-xr-x   39 blyth  staff  1248 Nov 12  2022 ..
    0 drwxr-xr-x    8 blyth  staff   256 Oct  2  2022 GDXML
    0 drwxr-xr-x    4 blyth  staff   128 Oct  2  2022 GeoChain
    0 drwxr-xr-x    4 blyth  staff   128 Oct  1  2022 UseFindOpticks
    0 drwxr-xr-x  110 blyth  staff  3520 May 18  2022 CFG4
    0 drwxr-xr-x   39 blyth  staff  1248 Apr 16  2022 OptiXRap
    0 drwxr-xr-x   11 blyth  staff   352 Apr  1  2022 G4OK
    0 drwxr-xr-x    7 blyth  staff   224 Apr  1  2022 OKG4
    0 drwxr-xr-x    8 blyth  staff   256 Apr  1  2022 OK
    0 drwxr-xr-x   10 blyth  staff   320 Apr  1  2022 OpticksGL
    0 drwxr-xr-x   33 blyth  staff  1056 Apr  1  2022 OGLRap
    0 drwxr-xr-x   14 blyth  staff   448 Apr  1  2022 OKOP
    0 drwxr-xr-x   15 blyth  staff   480 Apr  1  2022 ThrustRap
    0 drwxr-xr-x   16 blyth  staff   512 Apr  1  2022 CUDARap
    0 drwxr-xr-x   10 blyth  staff   320 Apr  1  2022 OpticksGeo
    epsilon:include blyth$ rm -rf NPY CSG_GGeo PMTFastSim CSG_U4 OpticksCore BoostRap GeoChain UseFindOpticks CFG4 OptiXRap G4OK OKG4 OK OpticksGL OGLRap OKOP ThrustRap CUDARap OpticksGeo 
    epsilon:include blyth$ 





YEP CSGOptiX has similar issue::


    === om-make-one : CSGOptiX        /Users/blyth/opticks/CSGOptiX                                /usr/local/opticks/build/CSGOptiX                            
    -- Configuring CSGOptiX
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- Found boost_system 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_system-1.70.0
    --   libboost_system.dylib
    -- Adding boost_system dependencies: headers
    -- Found boost_headers 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_headers-1.70.0
    -- Found boost_program_options 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_program_options-1.70.0
    --   libboost_program_options.dylib
    -- Adding boost_program_options dependencies: headers
    -- Found boost_filesystem 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_filesystem-1.70.0
    --   libboost_filesystem.dylib
    -- Adding boost_filesystem dependencies: headers
    -- Found boost_regex 1.70.0 at /usr/local/opticks_externals/boost/lib/cmake/boost_regex-1.70.0
    --   libboost_regex.dylib
    -- Adding boost_regex dependencies: headers
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++11
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERBOSE : ON 
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_MODULE  : /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake 
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_INCLUDE : /usr/local/optix/include 
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : OpticksOptiX_VERSION : 50001 : is pre-7  
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : optix_LIBRARY       : /usr/local/optix/lib64/liboptix.1.dylib 
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : optixu_LIBRARY      : /usr/local/optix/lib64/liboptixu.1.dylib 
    -- /Users/blyth/opticks/cmake/Modules/FindOpticksOptiX.cmake : optix_prime_LIBRARY : /usr/local/optix/lib64/liboptix_prime.1.dylib 
    -- CSG_FOUND     : 1 
    -- CSG_INCLUDE_DIRS : /usr/local/opticks/include/CSG;/usr/local/opticks/include/CSG 
    -- OpticksOptiX_VERSION  : 50001 
    -- write to buildenvpath /usr/local/opticks/build/CSGOptiX/buildenv.sh 
    -- CU_SOURCES : CSGOptiX6.cu;CSGOptiX6geo.cu 
    -- _generated_OBJ_files 
    -- _generated_PTX_files /usr/local/opticks/build/CSGOptiX/CSGOptiX_generated_CSGOptiX6.cu.ptx;/usr/local/opticks/build/CSGOptiX/CSGOptiX_generated_CSGOptiX6geo.cu.ptx
    -- bcm_auto_pkgconfig_each LIB:Threads::Threads : MISSING LIB_PKGCONFIG_NAME 
    -- Configuring done
    CMake Error in CMakeLists.txt:
      Imported target "Opticks::OpticksCore" includes non-existent path

        "/usr/local/opticks/include/OpticksCore"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.



    CMake Error in CMakeLists.txt:
      Imported target "Opticks::OpticksCore" includes non-existent path

        "/usr/local/opticks/include/OpticksCore"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.



    CMake Error in CMakeLists.txt:
      Imported target "Opticks::OpticksCore" includes non-existent path

        "/usr/local/opticks/include/OpticksCore"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.



    -- Generating done
    CMake Generate step failed.  Build files cannot be regenerated correctly.
    make: *** [cmake_check_build_system] Error 1
    === om-one-or-all make : non-zero rc 2
    === om-all om-make : ERROR bdir /usr/local/opticks/build/CSGOptiX : non-zero rc 2
    === om-one-or-all make : non-zero rc 2
    epsilon:opticks blyth$ 


::

    epsilon:opticks blyth$ find . -name CMakeLists.txt -exec grep -H Opticks::OpticksCore {} \; 
    ./CSGOptiX/CMakeLists.txt:         Opticks::OpticksCore
    ./CSG/tests/CMakeLists.txt:    #target_link_libraries(${TGT} Opticks::CSG Opticks::OpticksCore)
    ./CSG/tests/CMakeLists.txt:#target_link_libraries(${TGT} Opticks::CSG Opticks::OpticksCore)
    ./opticksgeo/CMakeLists.txt:target_link_libraries( ${name} PUBLIC  Opticks::OpticksCore )
    ./ggeo/CMakeLists.txt:target_link_libraries( ${name} PUBLIC Opticks::OpticksCore)
    ./thrustrap/CMakeLists.txt:target_link_libraries( ${name} Opticks::OpticksCore Opticks::CUDARap)
    ./examples/UseOpticksCore/CMakeLists.txt:target_link_libraries(${name} Opticks::OpticksCore)
    epsilon:opticks blyth$ 



g4cx has another stray::


    === om-make-one : g4cx            /Users/blyth/opticks/g4cx                                    /usr/local/opticks/build/g4cx                                
    -- Configuring G4CX
    ...
    -- Configuring done
    CMake Error in CMakeLists.txt:
      Imported target "Opticks::ExtG4" includes non-existent path

        "/usr/local/opticks/include/OpticksCore"

      in its INTERFACE_INCLUDE_DIRECTORIES.  Possible reasons include:

      * The path was deleted, renamed, or moved to another location.

      * An install or uninstall procedure did not complete successfully.

      * The installation package was faulty and references files it does not
      provide.



Standardize clean build testing under francis
------------------------------------------------

::

    epsilon:~ francis$ cat deepclean_except_externals.sh 
    #!/bin/bash -l 

    usage(){ cat << EOU
    deepclean_except_externals.sh
    ==============================

    For testing almost from scratch builds of 
    opticks except keeping the externals : which 
    are prone to failure from network blockages. 

    EOU
    }

    rm -rf local/opticks/bin
    rm -rf local/opticks/build
    rm -rf local/opticks/include
    rm -rf local/opticks/lib
    rm -rf local/opticks/opticksaux
    rm -rf local/opticks/py

    epsilon:~ francis$ 


Hence test opticks-full as user francis with::

    ssh F

    # uses symbolic link to /Users/blyth/opticks so no clone
    ./deepclean_except_externals.sh
    opticks-full 


Above is laptop test.


