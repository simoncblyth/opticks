opticks-configure-optix-location
===================================


finding OptiX ?
-------------------

::

    Hi Simon

    Thanks for the update.
    I try to use the instruction you provide to fix the issue.
    But I still gets errors.

    It seems like even though I specify the path to the directory where I install OptiX, it seems still not finding the OptiX.
    Moreover the optic-compute-capability shows 0 

    I went back to reinstall the OptiX3.8, but still have the same error.

    The attached file is the output message from command line, this one should be small.

    BTW, We would like to buy GPU for our cluster here for developing the code.
    Do you have any suggestion that what GPU you would recommend ?

    Thank you

    Ryan


Observations
--------------

* *opticks-cmake-info* may be misleading as its describing the input, not what gets into the cmake cache

* *opticks-cmakecache-vars* dumps what cmake actually has in its cache

::

    simon:opticks blyth$ opticks-cmakecache-vars
    CMAKE_BUILD_TYPE:STRING=Debug
    COMPUTE_CAPABILITY:STRING=30
    CMAKE_INSTALL_PREFIX:PATH=/usr/local/opticks
    OptiX_INSTALL_DIR:PATH=/Developer/OptiX
    Geant4_DIR:UNINITIALIZED=/usr/local/opticks/externals/lib/Geant4-10.2.1
    XERCESC_LIBRARY:FILEPATH=/opt/local/lib/libxerces-c.dylib
    //ADVANCED property for variable: XERCESC_LIBRARY
    XERCESC_LIBRARY-ADVANCED:INTERNAL=1
    XERCESC_INCLUDE_DIR:PATH=/opt/local/include
    //ADVANCED property for variable: XERCESC_INCLUDE_DIR
    XERCESC_INCLUDE_DIR-ADVANCED:INTERNAL=1
    simon:opticks blyth$ 


* *opticks-cmake-modify-ex3* is an example of modifying a cmake configuration giving the OptiX_INSTALL_DIR and COMPUTE_CAPABILITY


::

    opticks-cmake-modify-ex3(){

      local msg="=== $FUNCNAME : "
      local bdir=$(opticks-bdir)
      local bcache=$bdir/CMakeCache.txt
      [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
      opticks-bcd

      echo $msg opticks-cmakecache-vars BEFORE MODIFY 
      opticks-cmakecache-vars 

      cmake \
           -DOptiX_INSTALL_DIR=/Developer/OptiX \
           -DCOMPUTE_CAPABILITY=30 \
              .   

      echo $msg opticks-cmakecache-vars AFTER MODIFY 
      opticks-cmakecache-vars 

    }






Log
-----

::


    Jui-Jens-MacBook-Pro:opticks wangbtc$ opticks-configure -DOptiX_INSTALL_DIR=/Developer/OptiX/
    === opticks-wipe : wiping build dir /usr/local/opticks/build
    === opticks-cmake : configuring installation

    opticks-cmake-info
    ======================

           opticks-sdir               :  /Users/wangbtc/opticks
           opticks-bdir               :  /usr/local/opticks/build
           opticks-cmake-generator    :  Unix Makefiles
           opticks-compute-capability :  0
           opticks-prefix             :  /usr/local/opticks
           opticks-optix-install-dir  :  /tmp
           g4-cmake-dir               :  /usr/local/opticks/externals/lib/Geant4-10.2.1
           xercesc-library            :  /opt/local/lib/libxerces-c.dylib
           xercesc-include-dir        :  /opt/local/include

    -- The C compiler identification is AppleClang 7.3.0.7030031
    -- The CXX compiler identification is AppleClang 7.3.0.7030031
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
    CMAKE_CXX_FLAGS = -W -Wall -pedantic -Wno-non-virtual-dtor -Wno-long-long -Wwrite-strings -Wpointer-arith -Woverloaded-virtual -Wno-variadic-macros -Wshadow -pipe -Qunused-arguments -stdlib=libc++
    CMAKE_CXX_FLAGS_DEBUG = -g -DG4FPE_DEBUG
    CMAKE_CXX_FLAGS_RELEASE = -O2 -DNDEBUG
    CMAKE_CXX_FLAGS_RELWITHDEBINFO= -O2 -g
    -- Boost version: 1.59.0
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
    NPY.DEFINITIONS : -DBOOST_LOG_DYN_LINK;-DWITH_YoctoGL;-DWITH_CSGBSP
    NPY.CSGBSP_FOUND
    NPY.YoctoGL_FOUND
    -- Configuring OpticksCore
    -- Configuring GGeo
    GGEO.NPY_INCLUDE_DIRS : /Users/wangbtc/opticks/opticksnpy
    GGEO.YoctoGL_FOUND
    GGEO.DEFINITIONS : -DBOOST_LOG_DYN_LINK;-DWITH_YoctoGL
    -- Configuring AssimpRap
    -- Configuring OpenMeshRap
    -- Configuring OpticksGeometry
    -- Configuring OGLRap
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
    -- Generating done
    -- Build files have been written to: /usr/local/opticks/build
    Jui-Jens-MacBook-Pro:opticks wangbtc$ ls /Developer/OptiX/
    SDK                     SDK-precompiled-samples build_1                 doc                     include                 lib64
    Jui-Jens-MacBook-Pro:opticks wangbtc$




