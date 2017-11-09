OpticksBufferSpec Undefined Symbols
=====================================

Symptom : Undefined symbols whilst linking libOpticksCore
-------------------------------------------------------------

::

    Undefined symbols for architecture x86_64:
      "OpticksBufferSpec::photon_compute_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::photon_interop_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::source_compute_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::source_interop_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::genstep_compute_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o
      "OpticksBufferSpec::genstep_interop_", referenced from:
          OpticksBufferSpec::Get(char const*, bool) in OpticksBufferSpec.cc.o


Known Causes
---------------

This issue occurs when 

1. the Opticks build fails to find OptiX 
2. OptiX is found but the version is not accounted for in the 
   preprocessor macros of optickscore/OpticksBufferSpec.cc


OptiX Not Found
~~~~~~~~~~~~~~~~~~~


If OptiX is not found you get something like::

    OGLRap.ImGui_INCLUDE_DIRS : /usr/local/opticks/externals/include
    -- Opticks.COMPUTE_CAPABILITY : 0 : at least 30 is required for Opticks, proceeding GPU-less
    -- Configuring OK
    Operating without OPTIX
    -- Configuring cfg4
    -- cfg4._line #define G4VERSION_NUMBER  952 ===> 952
    cfg4.XERCESC_INCLUDE_DIR  : /opt/local/include
    cfg4.XERCESC_LIBRARIES    : /opt/local/lib/libxerces-c.dylib
    -- Configuring okg4
    Operating without OPTIX
    top.OXRAP_OPTIX_VERSION
    CMAKE_INSTALL_PREFIX:/usr/local/opticks
    CMAKE_INSTALL_BINDIR:
    -- Configuring done


When OptiX is found you should see something like::

    -- Configuring OptiXRap
    -- OptiXRap.OPTICKS_OPTIX_VERSION : 3.8  FOUND 
    -- OptiXRap.OptiX_INCLUDE_DIRS    : /Developer/OptiX_380/include 
    -- OptiXRap.OptiX_LIBRARIES       : /Developer/OptiX_380/lib64/liboptix.dylib;/Developer/OptiX_380/lib64/liboptixu.dylib 
    -- OptiXRap._line #define OPTIX_VERSION 3080 /* 3.8.0 (major =  OPTIX_VERSION/1000,       * ===> 3080 
    -- Configuring OKOP
    -- Configuring OpticksGL
    -- Opticks.OXRAP_OPTIX_VERSION : 3080 
    -- Configuring OK
    -- Configuring cfg4



Preprocessor macro OXRAP_OPTIX_VERSION
-----------------------------------------

The macro OXRAP_OPTIX_VERSION is defined at CMake level, by parsing the OptiX header with CMake 

::

    simon:optickscore blyth$ opticks-find OXRAP_OPTIX_VERSION
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101 
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101
    ./optickscore/OpticksBufferSpec.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./optickscore/OpticksBufferSpec.cc:#elif OXRAP_OPTIX_VERSION == 400000 || OXRAP_OPTIX_VERSION == 40000 ||  OXRAP_OPTIX_VERSION == 40101
    ./optickscore/tests/OpticksBufferSpecTest.cc:    LOG(info) << "OXRAP_OPTIX_VERSION : " << OXRAP_OPTIX_VERSION ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION >= 3080
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION >= 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION >= 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION == 3080
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3080 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 3090
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3090 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 40000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 40000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 400000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION == 3080,3090,4000,400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#if OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 40000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 40000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:#elif OXRAP_OPTIX_VERSION == 400000
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " OXRAP_OPTIX_VERSION == 400000 : " << OXRAP_OPTIX_VERSION  ;
    ./sysrap/tests/OpticksCMakeConfigTest.cc:    LOG(info) << " (NOT) OXRAP_OPTIX_VERSION == 3080,3090,4000,400000 : " << OXRAP_OPTIX_VERSION  ;
    ./CMakeLists.txt:       message(STATUS "${name}.OXRAP_OPTIX_VERSION : ${OXRAP_OPTIX_VERSION} ")
    ./CMakeLists.txt:# collects version defines, currently only OXRAP_OPTIX_VERSION and CFG4_G4VERSION_NUMBER
    ./CMakeLists.txt:message("top.OXRAP_OPTIX_VERSION ${OXRAP_OPTIX_VERSION} ")
    ./optixrap/CMakeLists.txt:set(OXRAP_OPTIX_VERSION 0 PARENT_SCOPE)
    ./optixrap/CMakeLists.txt:            set(OXRAP_OPTIX_VERSION ${CMAKE_MATCH_1} PARENT_SCOPE)
    simon:opticks blyth$ 
    simon:opticks blyth$ 



What Version of OptiX was found ?
-----------------------------------

The first sysrap package creates a test executable **OpticksCMakeConfigTest**
that dumps the detected *OXRAP_OPTIX_VERSION*.

::

    simon:opticks blyth$ OpticksCMakeConfigTest
    2017-11-09 15:08:44.745 INFO  [3903455] [main@10]  OXRAP_OPTIX_VERSION >= 3080 : 3080
    2017-11-09 15:08:44.745 INFO  [3903455] [main@17]  OXRAP_OPTIX_VERSION == 3080 : 3080
    2017-11-09 15:08:44.746 INFO  [3903455] [main@32]  OXRAP_OPTIX_VERSION == 3080 || OXRAP_OPTIX_VERSION == 3090 : 3080
    simon:opticks blyth$ 
    simon:opticks blyth$ 



Configuring Opticks to find OptiX
-----------------------------------

Opticks needs to be told where to find the OptiX directory, 
the content of which should be similar to the below::

    simon:issues blyth$ l /Developer/OptiX_380/
    total 0
    drwxr-xr-x  90 root  admin  3060 Jun 29  2015 SDK-precompiled-samples
    drwxr-xr-x  37 root  admin  1258 Jun 29  2015 lib64
    drwxr-xr-x  65 root  admin  2210 May 29  2015 SDK
    drwxr-xr-x  17 root  admin   578 May 29  2015 include
    drwxr-xr-x   9 root  admin   306 May 29  2015 doc


Configuring will wipe the build directory, forcing a full build.   
There are alternatives for experts willing to study the opticks- bash functions.
Use *opticks-vi* to take a look at them.

Configuring the location of OptiX::


    simon:boostrap blyth$ opticks-configure -DOptiX_INSTALL_DIR=/Developer/OptiX_380
    === opticks-wipe : wiping build dir /usr/local/opticks/build
    === opticks-cmake : configuring installation

    opticks-cmake-info
    ======================

           opticks-sdir               :  /Users/blyth/opticks
           opticks-bdir               :  /usr/local/opticks/build
           opticks-cmake-generator    :  Unix Makefiles
           opticks-compute-capability :  30
           opticks-prefix             :  /usr/local/opticks
           opticks-optix-install-dir  :  /Developer/OptiX_380
           g4-cmake-dir               :  /usr/local/opticks/externals/lib/Geant4-10.2.1
           xercesc-library            :  /opt/local/lib/libxerces-c.dylib
           xercesc-include-dir        :  /opt/local/include

    -- The C compiler identification is AppleClang 6.0.0.6000057
    -- The CXX compiler identification is AppleClang 6.0.0.6000057
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc -- works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ -- works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Configuring Opticks
    CMAKE_BUILD_TYPE = Debug
    CMAKE_CXX_FLAGS =  -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -Qunused-arguments -stdlib=libc++
    CMAKE_CXX_FLAGS_DEBUG = -g -DG4FPE_DEBUG
    CMAKE_CXX_FLAGS_RELEASE = -O2 -DNDEBUG
    CMAKE_CXX_FLAGS_RELWITHDEBINFO= -O2 -g
    -- Boost version: 1.57.0
    -- Found the following Boost libraries:
    --   system
    --   program_options
    --   filesystem
    --   regex
    -- Configuring SysRap
    SysRap:CMAKE_BINARY_DIR : /usr/local/opticks/build 
    -- Configuring BoostRap
    -- Configuring NPY
    NPY.OpenMesh_LIBRARIES :/usr/local/opticks/externals/lib/libOpenMeshCore.dylib;/usr/local/opticks/externals/lib/libOpenMeshTools.dylib 
    NPY.CSGBSP_INCLUDE_DIRS:/usr/local/opticks/externals/csgbsp/csgjs-cpp 
    NPY.DEFINITIONS : -DBOOST_LOG_DYN_LINK;-DWITH_ImplicitMesher;-DWITH_DualContouringSample;-DWITH_YoctoGL;-DWITH_CSGBSP 
    NPY.ImplicitMesher_FOUND
    NPY.DualContouringSample_FOUND
    NPY.CSGBSP_FOUND
    NPY.YoctoGL_FOUND
    -- Configuring OpticksCore
    -- Configuring GGeo
    GGEO.NPY_INCLUDE_DIRS : /Users/blyth/opticks/opticksnpy
    GGEO.YoctoGL_FOUND
    -- Configuring AssimpRap
    -- Configuring OpenMeshRap
    -- Configuring OpticksGeometry
    -- Configuring OGLRap
    OGLRap.ImGui_INCLUDE_DIRS : /usr/local/opticks/externals/include
    -- Opticks.COMPUTE_CAPABILITY : 30
    -- Opticks.CUDA_NVCC_FLAGS    : -Xcompiler -fPIC;-gencode=arch=compute_30,code=sm_30;-std=c++11;-O2;--use_fast_math 
    -- Looking for pthread.h
    -- Looking for pthread.h - found
    -- Looking for pthread_create
    -- Looking for pthread_create - found
    -- Found Threads: TRUE  
    -- Found CUDA: /usr/local/cuda (found suitable version "7.0", minimum required is "7.0") 
    -- Configuring CUDARap
    -- Configuring ThrustRap
    -- ThrustRap.CUDA_NVCC_FLAGS : 
    -- Configuring OptiXRap
    -- OptiXRap.OPTICKS_OPTIX_VERSION : 3.8  FOUND 
    -- OptiXRap.OptiX_INCLUDE_DIRS    : /Developer/OptiX_380/include 
    -- OptiXRap.OptiX_LIBRARIES       : /Developer/OptiX_380/lib64/liboptix.dylib;/Developer/OptiX_380/lib64/liboptixu.dylib 
    -- OptiXRap._line #define OPTIX_VERSION 3080 /* 3.8.0 (major =  OPTIX_VERSION/1000,       * ===> 3080 
    -- Checking to see if CXX compiler accepts flag -Wno-unused-result
    -- Checking to see if CXX compiler accepts flag -Wno-unused-result - yes
    -- Performing Test SSE_41_AVAILABLE
    -- Performing Test SSE_41_AVAILABLE - Success
    -- Configuring OKOP
    -- Configuring OpticksGL
    -- Opticks.OXRAP_OPTIX_VERSION : 3080 
    -- Configuring OK
    -- Configuring cfg4
    -- cfg4._line #define G4VERSION_NUMBER  1021 ===> 1021 
    cfg4.XERCESC_INCLUDE_DIR  : /opt/local/include 
    cfg4.XERCESC_LIBRARIES    : /opt/local/lib/libxerces-c.dylib 
    -- Configuring okg4
    top.OXRAP_OPTIX_VERSION 3080 
    CMAKE_INSTALL_PREFIX:/usr/local/opticks 
    CMAKE_INSTALL_BINDIR: 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /usr/local/opticks/build
    simon:boostrap blyth$ 




