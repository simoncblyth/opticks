Two out of 22 libs CUDARap.so and libThrustRap.so are duplicated into lib and lib64 : FIXED
=============================================================================================

The duplication was due to an extraneous "install" into hardcoded *lib* 
forgetting that bcm_deploy does that already and uses the non-hardcoded
CMAKE_INSTALL_LIBDIR 


Issue : duplicate install of two libs
----------------------------------------

::

    blyth@localhost boostrap]$ ll /home/blyth/local/opticks/lib64/
    total 106524
    drwxrwxr-x. 12 blyth blyth      152 Apr  1 18:19 ..
    drwxrwxr-x. 31 blyth blyth     4096 Apr  9 15:42 cmake
    -rwxr-xr-x.  1 blyth blyth   662600 Apr  9 19:48 libUseCUDA.so
    -rwxr-xr-x.  1 blyth blyth    32368 Apr  9 21:16 libOKConf.so
    -rwxr-xr-x.  1 blyth blyth  1123400 Apr  9 21:16 libSysRap.so
    -rwxr-xr-x.  1 blyth blyth 11327648 Apr  9 21:25 libBoostRap.so
    -rwxr-xr-x.  1 blyth blyth 25732864 Apr  9 21:26 libNPY.so
    -rwxr-xr-x.  1 blyth blyth  2594560 Apr  9 21:26 libYoctoGLRap.so
    -rwxr-xr-x.  1 blyth blyth  9924632 Apr  9 21:26 libOpticksCore.so
    -rwxr-xr-x.  1 blyth blyth 10237960 Apr  9 21:26 libGGeo.so
    -rwxr-xr-x.  1 blyth blyth  1164464 Apr  9 21:26 libAssimpRap.so
    -rwxr-xr-x.  1 blyth blyth  2734968 Apr  9 21:26 libOpenMeshRap.so
    -rwxr-xr-x.  1 blyth blyth  1363136 Apr  9 21:26 libOpticksGeo.so
    -rwxr-xr-x.  1 blyth blyth  1256976 Apr  9 21:27 libCUDARap.so
    -rwxr-xr-x.  1 blyth blyth  3866064 Apr  9 21:27 libThrustRap.so
    -rwxr-xr-x.  1 blyth blyth  4791744 Apr  9 21:27 libOptiXRap.so
    -rwxr-xr-x.  1 blyth blyth  2507880 Apr  9 21:28 libOKOP.so
    -rwxr-xr-x.  1 blyth blyth  4776888 Apr  9 21:28 libOGLRap.so
    -rwxr-xr-x.  1 blyth blyth   737640 Apr  9 21:28 libOpticksGL.so
    -rwxr-xr-x.  1 blyth blyth   367376 Apr  9 21:28 libOK.so
    -rwxr-xr-x.  1 blyth blyth  4171728 Apr  9 21:28 libExtG4.so
    -rwxr-xr-x.  1 blyth blyth 17926032 Apr  9 21:28 libCFG4.so
    -rwxr-xr-x.  1 blyth blyth   293864 Apr  9 21:28 libOKG4.so
    -rwxr-xr-x.  1 blyth blyth   705928 Apr  9 21:28 libG4OK.so
    drwxrwxr-x.  4 blyth blyth     4096 Apr  9 21:28 .
    drwxrwxr-x.  2 blyth blyth     4096 Apr  9 21:28 pkgconfig
    [blyth@localhost boostrap]$ 
    [blyth@localhost boostrap]$ cd ..
    [blyth@localhost opticks]$ ll /home/blyth/local/opticks/lib/*.so
    -rwxr-xr-x. 1 blyth blyth 1256976 Apr  9 21:27 /home/blyth/local/opticks/lib/libCUDARap.so
    -rwxr-xr-x. 1 blyth blyth 3866064 Apr  9 21:27 /home/blyth/local/opticks/lib/libThrustRap.so

::

    [blyth@localhost opticks]$ diff lib/libCUDARap.so lib64/libCUDARap.so 
    [blyth@localhost opticks]$ diff lib/libThrustRap.so lib64/libThrustRap.so 



Deleting lib/libCUDARap.so and rebuilding::

   cudarap-
   cudarap-c
   om-clean
   om-conf
   om-make

Recreates into lib and copies it into lib64::

    ...
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /home/blyth/local/opticks/lib/libCUDARap.so
    -- Set runtime path of "/home/blyth/local/opticks/lib/libCUDARap.so" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CUDARAP_LOG.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CUDARAP_API_EXPORT.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CUDARAP_HEAD.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CUDARAP_TAIL.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/LaunchCommon.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/LaunchSequence.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/cuRANDWrapper.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/cuRANDWrapper_kernel.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CResource.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CBufSpec.hh
    -- Up-to-date: /home/blyth/local/opticks/include/CUDARap/CBufSlice.hh
    -- Installing: /home/blyth/local/opticks/lib64/libCUDARap.so
    -- Set runtime path of "/home/blyth/local/opticks/lib64/libCUDARap.so" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Installing: /home/blyth/local/opticks/lib64/pkgconfig/cudarap.pc
    -- Installing: /home/blyth/local/opticks/lib64/cmake/cudarap/properties-cudarap-targets.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/cudarap/cudarap-targets.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/cudarap/cudarap-targets-debug.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/cudarap/cudarap-config.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/cudarap/cudarap-config-version.cmake
    -- Installing: /home/blyth/local/opticks/lib/LaunchSequenceTest
    -- Set runtime path of "/home/blyth/local/opticks/lib/LaunchSequenceTest" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Installing: /home/blyth/local/opticks/lib/cuRANDWrapperTest
    -- Set runtime path of "/home/blyth/local/opticks/lib/cuRANDWrapperTest" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Installing: /home/blyth/local/opticks/lib/curand_aligned_host
    -- Set runtime path of "/home/blyth/local/opticks/lib/curand_aligned_host" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Installing: /home/blyth/local/opticks/lib/CUDARapVersionTest
    -- Set runtime path of "/home/blyth/local/opticks/lib/CUDARapVersionTest" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"
    -- Installing: /home/blyth/local/opticks/lib/cudaGetDevicePropertiesTest
    -- Set runtime path of "/home/blyth/local/opticks/lib/cudaGetDevicePropertiesTest" to "/home/blyth/local/opticks/lib64:/usr/local/cuda-10.1/lib64"



    [blyth@localhost cudarap]$ ll /home/blyth/local/opticks/lib/libCUDARap.so /home/blyth/local/opticks/lib64/libCUDARap.so
    -rwxr-xr-x. 1 blyth blyth 1256976 Apr 10 10:07 /home/blyth/local/opticks/lib/libCUDARap.so
    -rwxr-xr-x. 1 blyth blyth 1256976 Apr 10 10:07 /home/blyth/local/opticks/lib64/libCUDARap.so



Possible from CUDA_ADD_LIBRARY effect : NOPE
-------------------------------------------------

cudarap and thrustrap are the only mainline subs using CUDA_ADD_LIBRARY::

    [blyth@localhost opticks]$ find . -name 'CMakeLists.txt' -exec grep -H CUDA_ADD_LIBRARY {} \;
    ./examples/ThrustRapMinimal/CMakeLists.txt:message(STATUS "${name} CUDA_ADD_LIBRARY.INTERFACE_LINK_LIBRARIES : ${_cal_ill} ") 
    ./examples/ThrustRapMinimal/CMakeLists.txt:-- ThrustRap CUDA_ADD_LIBRARY.INTERFACE_LINK_LIBRARIES : /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a;-Wl,-rpath,/usr/local/cuda/lib 
    ./examples/CMakeLists.txt:     56 CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
    ./examples/UseCUDA/CMakeLists.txt:CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
    ./examples/UseCUDA/CMakeLists.txt:# CUDA_ADD_LIBRARY sets raw library path and linker args into target prop INTERFACE_LINK_LIBRARIES
    ./cudarap/CMakeLists.txt:CUDA_ADD_LIBRARY( ${name} ${SOURCES} )
    ./thrustrap/CMakeLists.txt:CUDA_ADD_LIBRARY( ${name} ${SOURCES} OPTIONS )
    ./thrustrap/CMakeLists.txt:-- ThrustRap CUDA_ADD_LIBRARY.INTERFACE_LINK_LIBRARIES : /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a;-Wl,-rpath,/usr/local/cuda/lib 
    ./thrustrap/CMakeLists.txt:   message(STATUS "${name} CUDA_ADD_LIBRARY.INTERFACE_LINK_LIBRARIES : ${_cal_ill} ") 
    [blyth@localhost opticks]$ 

But UseCUDA also does CUDA_ADD_LIBRARY and it does not install into lib, examples/UseCUDA/go.sh::

    Install the project...
    -- Install configuration: "Debug"
    -- Up-to-date: /home/blyth/local/opticks/include/UseCUDA/UseCUDA.h
    -- Installing: /home/blyth/local/opticks/lib64/libUseCUDA.so
    -- Installing: /home/blyth/local/opticks/lib64/pkgconfig/usecuda.pc
    -- Installing: /home/blyth/local/opticks/lib64/cmake/usecuda/properties-usecuda-targets.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/usecuda/usecuda-targets.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/usecuda/usecuda-targets-debug.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/usecuda/usecuda-config.cmake
    -- Installing: /home/blyth/local/opticks/lib64/cmake/usecuda/usecuda-config-version.cmake
    [blyth@localhost UseCUDA]$ 

cudarap/CMakeLists.txt::

     75 install(TARGETS ${name} LIBRARY DESTINATION lib)
     76 install(FILES ${HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
     77 
     78 bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
     79 


Other projects dont install to *lib* they defer to bcm_deploy.

sysrap/CMakeLists.txt::

    115 
    116 bcm_deploy(TARGETS ${name} NAMESPACE Opticks:: SKIP_HEADER_INSTALL)
    117 install(FILES ${HEADERS}  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    118 


See *bcm-vi* for a review of what bcm does.







