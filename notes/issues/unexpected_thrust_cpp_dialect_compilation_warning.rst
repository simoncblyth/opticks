unexpected_thrust_cpp_dialect_compilation_warning
====================================================


Overview
---------

CSGOptiX/CMakeLists.txt was resetting the flags from cmake/Modules/OpticksCUDAFlags.cmake::

    #set(COMPUTE_CAPABILITY $ENV{OPTICKS_COMPUTE_CAPABILITY})
    #set(CUDA_NVCC_FLAGS)
    #list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
    #list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
    #list(APPEND CUDA_NVCC_FLAGS "-O2")
    #list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
    #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")

    #[=[
    CUDA_NVCC_FLAGS are set in cmake/Modules/OpticksCUDAFlags.cmake
    its better to set them in one place
    #]=]


Issue
--------

::

    -- Build files have been written to: /data/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX
    [  5%] Building NVCC ptx file CSGOptiX_generated_Check.cu.ptx
    [  5%] Building NVCC ptx file CSGOptiX_generated_CSGOptiX7.cu.ptx
    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/complex.h:24,
                     from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/include/Custom4/C4MultiLayrStack.h:59,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h:29,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h:66,
                     from /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX7.cu:70:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                                                                           

::

    -- Configuring QUDARap
    -- cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable c++17
    -- OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : 70
    -- OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : -Xcompiler -fPIC;-gencode=arch=compute_70,code=sm_70;-std=c++17;-O2;--use_fast_math;-Xcudafe --diag_suppress=esa_on_defaulted_function_ignored  



::

    N[blyth@localhost CSGOptiX]$ touch CSGOptiX7.cu 
    N[blyth@localhost CSGOptiX]$ om
    === om-env : normal running
    === opticks-setup-       skip     append                 PATH /usr/local/cuda-11.7/bin
    === opticks-setup-       skip     append                 PATH /data/blyth/junotop/ExternalLibs/opticks/head/bin
    === opticks-setup-       skip     append                 PATH /data/blyth/junotop/ExternalLibs/opticks/head/lib
    === opticks-setup-       skip     append    CMAKE_PREFIX_PATH /data/blyth/junotop/ExternalLibs/opticks/head
    === opticks-setup-       skip     append    CMAKE_PREFIX_PATH /data/blyth/junotop/ExternalLibs/opticks/head/externals
    === opticks-setup-       skip     append    CMAKE_PREFIX_PATH /home/blyth/local/opticks/externals/OptiX_750
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /data/blyth/junotop/ExternalLibs/opticks/head/lib/pkgconfig
    === opticks-setup-       skip     append      PKG_CONFIG_PATH /data/blyth/junotop/ExternalLibs/opticks/head/lib64/pkgconfig
    === opticks-setup-       skip     append      PKG_CONFIG_PATH /data/blyth/junotop/ExternalLibs/opticks/head/externals/lib/pkgconfig
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /data/blyth/junotop/ExternalLibs/opticks/head/externals/lib64/pkgconfig
    === opticks-setup-       skip     append      LD_LIBRARY_PATH /data/blyth/junotop/ExternalLibs/opticks/head/lib
    === opticks-setup-       skip     append      LD_LIBRARY_PATH /data/blyth/junotop/ExternalLibs/opticks/head/lib64
    === opticks-setup-       skip     append      LD_LIBRARY_PATH /data/blyth/junotop/ExternalLibs/opticks/head/externals/lib
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /data/blyth/junotop/ExternalLibs/opticks/head/externals/lib64
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /usr/local/cuda-11.7/lib
    === opticks-setup-       skip     append      LD_LIBRARY_PATH /usr/local/cuda-11.7/lib64
    === opticks-setup-geant4- : sourcing /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J22.2.x/ExternalLibs/Geant4/10.04.p02.juno/bin/geant4.sh
    === om-make-one : CSGOptiX        /data/blyth/junotop/opticks/CSGOptiX                         /data/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX 
    [  2%] Building NVCC ptx file CSGOptiX_generated_CSGOptiX7.cu.ptx
    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/complex.h:24,
                     from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/include/Custom4/C4MultiLayrStack.h:59,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h:29,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h:66,
                     from /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX7.cu:70:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                      




::

    N[blyth@localhost CSGOptiX]$ touch CSGOptiX7.cu 
    N[blyth@localhost CSGOptiX]$ export VERBOSE=1
    N[blyth@localhost CSGOptiX]$ om



    /usr/local/cuda-11.7/bin/nvcc
          -M 
          -D__CUDACC__ 
          /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX7.cu 
          -o 
          /data/blyth/junotop/ExternalLibs/opticks/head/build/CSGOptiX/CMakeFiles/CSGOptiX.dir//CSGOptiX_generated_CSGOptiX7.cu.ptx.NVCC-depend 
          -ccbin 
          /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/bin/gcc 
          -m64 
          -DWITH_PRD 
          -DWITH_SGLM 
          -DOPTICKS_CSGOPTIX 
          -DDEBUG_TAG 
          -DWITH_THRUST 
          -DOPTICKS_CSG 
          -DWITH_CONTIGUOUS 
          -DWITH_S_BB 
          -DWITH_CHILD 
          -DOPTICKS_SYSRAP 
          -DPLOG_LOCAL 
          -DWITH_STTF 
          -DWITH_SLOG 
          -DOPTICKS_OKCONF 
          -DWITH_CUSTOM4 
          -DOPTICKS_QUDARAP 
          -DDEBUG_PIDX 
          -Xcompiler 
          -fPIC 
          -gencode=arch=compute_70,code=sm_70 
          -O2 
          --use_fast_math 
          -std=c++11 
          -DNVCC 
          -I/usr/local/cuda-11.7/include 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/include/CSG 
          -I/home/blyth/local/opticks/externals/OptiX_750/include 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/externals/glm/glm 
          -I/data/blyth/junotop/opticks/CSGOptiX 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/include/SysRap 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/externals/plog/include 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/include/OKConf 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/externals/include/nljson 
          -I/cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/include/Custom4 
          -I/data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap

    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/complex.h:24,
                     from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/include/Custom4/C4MultiLayrStack.h:59,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h:29,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h:66,
                     from /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX7.cu:70:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                                                                           
    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/complex.h:24,
                     from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.0-rc3.dc1/ExternalLibs/custom4/0.1.8/include/Custom4/C4MultiLayrStack.h:59,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qpmt.h:29,
                     from /data/blyth/junotop/ExternalLibs/opticks/head/include/QUDARap/qsim.h:66,
                     from /data/blyth/junotop/opticks/CSGOptiX/CSGOptiX7.cu:70:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                                                                           


