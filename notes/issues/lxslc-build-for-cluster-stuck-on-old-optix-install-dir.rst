lxslc-build-for-cluster-stuck-on-old-optix-install-dir FIXED
=================================================================

Solution
---------

Solution is just to change habit: 

* USE **om-install** or **om-cleaninstall** 
* DO NOT use om-conf then om-make 
  
  * (although this usually works fine, it will fail on changing configuration)

Issue
-------

Even following nuclear option of deleting build dir, the oxrap config
step from **om-conf** still getting the old OptiX_INSTALL_DIR ?

* thats because okconf needs to be conf, built and installed 
  before oxrap is conf in order to write okconf-config.cmake::

  /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-config.cmake

* been tripped up with this before, how to check ?

Theres a comment in om-usage::

   When building for the first time it is necessary to 
   use "om-install" as later subprojects cannot be configured 
   until earlier ones have been installed.

This is because::

    om-install-one()
    {
        om-visit-one $*
        om-conf-one $*
        om-make-one $*
    }


om-conf them om-- (om-make) showing the writing of the TOPMATTER into opticks/lib64/cmake/okconf/okconf-config.cmake
----------------------------------------------------------------------------------------------------------------------

om-conf
~~~~~~~~

::

    [blyth@lxslc701 okconf]$ om-conf
    === om-one-or-all conf : okconf          /afs/ihep.ac.cn/users/b/blyth/g/opticks/okconf               /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/okconf   
    -- Configuring OKConf
    -- OpticksCompilationFlags.cmake : CMAKE_BUILD_TYPE = Debug
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS =  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_DEBUG = -g
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELEASE = -O3 -DNDEBUG
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELWITHDEBINFO= -O2 -g -DNDEBUG
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD : 14 
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD_REQUIRED : on 
    -- Use examples/UseOpticksCUDA/CMakeLists.txt for testing FindOpticksCUDA.cmake
    --   CUDA_TOOLKIT_ROOT_DIR   : /usr/local/cuda 
    --   CUDA_SDK_ROOT_DIR       : CUDA_SDK_ROOT_DIR-NOTFOUND 
    --   CUDA_VERSION            : 10.1 
    --   HELPER_CUDA_INCLUDE_DIR : /usr/local/cuda/samples/common/inc 
    --   PROJECT_SOURCE_DIR      : /afs/ihep.ac.cn/users/b/blyth/g/opticks/okconf 
    --   CMAKE_CURRENT_LIST_DIR  : /afs/ihep.ac.cn/users/b/blyth/g/opticks/cmake/Modules 
    -- FindOpticksCUDA.cmake:OpticksCUDA_VERBOSE      : YES 
    -- FindOpticksCUDA.cmake:OpticksCUDA_FOUND        : YES 
    -- FindOpticksCUDA.cmake:OpticksHELPER_CUDA_FOUND : YES 
    -- FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION  : 10010 
    -- FindOpticksCUDA.cmake:CUDA_LIBRARIES           : /usr/local/cuda/lib64/libcudart_static.a;-lpthread;dl;/usr/lib64/librt.so 
    -- FindOpticksCUDA.cmake:CUDA_INCLUDE_DIRS        : /usr/local/cuda/include 
    -- FindOpticksCUDA.cmake:CUDA_curand_LIBRARY      : /usr/local/cuda/lib64/libcurand.so
     key='CUDA_cudart_static_LIBRARY' val='/usr/local/cuda/lib64/libcudart_static.a' 
     key='CUDA_curand_LIBRARY' val='/usr/local/cuda/lib64/libcurand.so' 

    -- /afs/ihep.ac.cn/users/b/blyth/g/opticks/cmake/Modules/FindOptiX.cmake : OptiX_VERBOSE     : ON 
    -- /afs/ihep.ac.cn/users/b/blyth/g/opticks/cmake/Modules/FindOptiX.cmake : OptiX_INSTALL_DIR : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX 
    -- FindOptiX.cmake.OptiX_MODULE          : /afs/ihep.ac.cn/users/b/blyth/g/opticks/cmake/Modules/FindOptiX.cmake
    -- FindOptiX.cmake.OptiX_FOUND           : YES
    -- FindOptiX.cmake.OptiX_VERSION_INTEGER : 60000
    -- FindOptiX.cmake.OptiX_INCLUDE         : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/include
    -- FindOptiX.cmake.optix_LIBRARY         : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/lib64/liboptix.so
    -- FindOptiX.cmake.optixu_LIBRARY        : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/lib64/liboptixu.so
    -- FindOptiX.cmake.optix_prime_LIBRARY   : /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX/lib64/liboptix_prime.so
    -- OKCONF_OPTIX_INSTALL_DIR : 
    -- OptiX_VERSION_INTEGER : 60000
    -- OpticksCUDA_API_VERSION : 10010
    -- G4_VERSION_INTEGER      : 1042
    -- Configuring OKConfTest
    -- target OKConf exists
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/okconf



om-make (om--)
~~~~~~~~~~~~~~~

::

    [blyth@lxslc701 okconf]$ om--
    === om-make-one : okconf          /afs/ihep.ac.cn/users/b/blyth/g/opticks/okconf               /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/build/okconf   
    Scanning dependencies of target OKConf
    [ 40%] Building CXX object CMakeFiles/OKConf.dir/OKConf.cc.o
    [ 40%] Building CXX object CMakeFiles/OKConf.dir/OpticksVersionNumber.cc.o
    [ 60%] Linking CXX shared library libOKConf.so
    [ 60%] Built target OKConf
    Scanning dependencies of target OKConfTest
    [ 80%] Building CXX object tests/CMakeFiles/OKConfTest.dir/OKConfTest.cc.o
    [100%] Linking CXX executable OKConfTest
    [100%] Built target OKConfTest
    [ 60%] Built target OKConf
    [100%] Built target OKConfTest
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/libOKConf.so
    -- Set runtime path of "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/libOKConf.so" to "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64"
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/pkgconfig/okconf.pc
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/properties-okconf-targets.cmake
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-targets.cmake
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-targets-debug.cmake
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-config.cmake
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-config-version.cmake
    -- Up-to-date: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/include/OKConf/OKConf.hh
    -- Up-to-date: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/include/OKConf/OpticksVersionNumber.hh
    -- Up-to-date: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/include/OKConf/OKCONF_API_EXPORT.hh
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/include/OKConf/OKConf_Config.hh
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib/OKConfTest
    -- Set runtime path of "/afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib/OKConfTest" to "$ORIGIN/../lib:$ORIGIN/../lib64:$ORIGIN/../externals/lib:$ORIGIN/../externals/lib64:$ORIGIN/../externals/OptiX/lib64"
    [blyth@lxslc701 okconf]$ 



opticks/lib64/cmake/okconf/okconf-config.cmake
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


::

    [blyth@lxslc701 okconf]$ cat /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/lib64/cmake/okconf/okconf-config.cmake

    # TOPMATTER

    ## OKConf generated TOPMATTER

    set(OptiX_INSTALL_DIR /afs/ihep.ac.cn/users/b/blyth/g/local/opticks/externals/OptiX)
    set(COMPUTE_CAPABILITY 70)

    if(OKConf_VERBOSE)
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OKConf_VERBOSE     : ${OKConf_VERBOSE} ")
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : OptiX_INSTALL_DIR  : ${OptiX_INSTALL_DIR} ")
      message(STATUS "${CMAKE_CURRENT_LIST_FILE} : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY} ")
    endif()

    include(OpticksCUDAFlags)

    ## see notes/issues/OpticksCUDAFlags.rst

    include(CMakeFindDependencyMacro)

    include("${CMAKE_CURRENT_LIST_DIR}/okconf-targets.cmake")
    include("${CMAKE_CURRENT_LIST_DIR}/properties-okconf-targets.cmake")
    [blyth@lxslc701 okconf]$ 



