optix-version-switching
===========================

scratch instructions
------------------------


1. set the override envvar in .bashrc:: 

    ## the location to look for OptiX defaults to $(opticks-prefix)/externals/OptiX
    ## to override that while testing another OptiX version set the below envvar 
    unset OPTICKS_OPTIX_INSTALL_DIR
    export OPTICKS_OPTIX_INSTALL_DIR=/usr/local/OptiX_511  ## override opticks-optix-install-dir 


2. start new shell, and do a clean OKConf reconfigure::

    opticks-   # skip if in your .bashrc already 
    okconf-    # define the okconf-* functions
    okconf---   # do a clean, config and build of OKConf 

    ## check that the CMake output shows the desired version and location


3. run OKConfTest executable to check get the expected versions 

::

    [blyth@localhost opticks]$ OKConfTest
    OKConf::Dump
                         OKConf::CUDAVersionInteger() 10010
                        OKConf::OptiXVersionInteger() 50101
                   OKConf::ComputeCapabilityInteger() 70
                            OKConf::CMAKE_CXX_FLAGS()  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow
                            OKConf::OptiXInstallDir() /usr/local/OptiX_511
                       OKConf::Geant4VersionInteger() 1042
                       OKConf::OpticksInstallPrefix() /home/blyth/local/opticks
                       OKConf::ShaderDir()            /home/blyth/local/opticks/gl

     OKConf::Check() 0


4. do full clean, config and build of Opticks

::

    cd ~/opticks
    om-clean
    om-conf
    om-make



notes
----------

Looking into making it easier to change optix version, notice 
that oxrap FindOptiX finds /usr/local/OptiX_600 but thats
not where OptiX_INSTALL_DIR : /home/blyth/local/opticks/externals/OptiX 
is pointed ?

::

    calhost opticks]$ oxrap--
    === om-make-one : optixrap        /home/blyth/opticks/optixrap                                 /home/blyth/local/opticks/build/optixrap                     
    ...
    -- Configuring OptiXRap
    -- /home/blyth/opticks/cmake/Modules/FindOptiX.cmake : OptiX_VERBOSE     : ON 
    -- /home/blyth/opticks/cmake/Modules/FindOptiX.cmake : OptiX_INSTALL_DIR : /home/blyth/local/opticks/externals/OptiX 
    -- FindOptiX.cmake.OptiX_MODULE          : /home/blyth/opticks/cmake/Modules/FindOptiX.cmake
    -- FindOptiX.cmake.OptiX_FOUND           : YES
    -- FindOptiX.cmake.OptiX_VERSION_INTEGER : 60000
    -- FindOptiX.cmake.OptiX_INCLUDE         : /usr/local/OptiX_600/include
    -- FindOptiX.cmake.optix_LIBRARY         : /usr/local/OptiX_600/lib64/liboptix.so
    -- FindOptiX.cmake.optixu_LIBRARY        : /usr/local/OptiX_600/lib64/liboptixu.so
    -- FindOptiX.cmake.optix_prime_LIBRARY   : /usr/local/OptiX_600/lib64/liboptix_prime.so
    -- Boost version: 1.53.0
    -- Found the following Boost libraries:
    --   system
    --   program_options


I recall, this is because OptiX finding is done by okconf (just to get the OptiX version from the 
header) which persists the location and is subsequently used by OptiXRap.   

Hmm this is confusing, maybe should tack OptiX target on to OKConf target ? Nope that means 
must find OptiX to build anything.

The reason for OKConf to find OptiX is so that all Opticks subprojs can know the
OptiX version at compile time via the generated header ~/local/opticks/include/OKConf/OKConf_Config.hh

::

     01 #pragma once
      2 
      3 //
      4 // First subproj OKConf CMakeLists.txt writes /home/blyth/local/opticks/build/okconf/inc/OpticksCMakeConfig.hh 
      5 // at configure time, based on version defines parsed from package headers 
      6 // in for example optixrap/CMakeLists.txt
      7 //
      8 // OpticksCMakeConfig.hh.in  -> OpticksCMakeConfig.hh 
      9 //
     10 // This means that package versions become globally available for all 
     11 // Opticks projects at all levels, without having to include package headers.
     12 //
     13 
     14 #define OKCONF_CUDA_API_VERSION_INTEGER 10010
     15 #define OKCONF_OPTIX_VERSION_INTEGER 60000
     16 #define OKCONF_GEANT4_VERSION_INTEGER 1042
     17 #define OKCONF_COMPUTE_CAPABILITY_INTEGER 70
     18 
     19 #define OKCONF_OPTICKS_INSTALL_PREFIX "/home/blyth/local/opticks"
     20 #define OKCONF_OPTIX_INSTALL_DIR "/home/blyth/local/opticks/externals/OptiX"
     21 /* #undef OKCONF_CUDA_NVCC_FLAGS */
     22 #define OKCONF_CMAKE_CXX_FLAGS   " -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow"
     23 
     24 
     25 // backward compat  TODO: adopt the new OKCONF names and remove these
     26 #define OXRAP_OPTIX_VERSION 60000
     27 #define CFG4_G4VERSION_NUMBER 1042
     28 




now compiles with Optix_511 but runtime needs hand holding to find libs
---------------------------------------------------------------------------

::

   blyth@localhost build]$ OKTest 
   OKTest: error while loading shared libraries: liboptix.so.51: cannot open shared object file: No such file or directory


   CUDA_VISIBLE_DEVICES=1 LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:/usr/local/OptiX_511/lib64 OKTest


This is because::

    - Set runtime path of "/home/blyth/local/opticks/lib/CX4GDMLTest" to "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"


Hmm regard this as penance for using a non-standard OptiX version 







