Versions Misc
================



CUDA 10.1 and gcc version constraints
----------------------------------------

The issue with Opticks and gcc 11.2 is that CUDA 10.1 
which I am currently using with OptiX 7.0.0
can only be used with up to gcc 8.

In order for Opticks to compile with gcc 11.2 
I will need several version updates::
   

    OptiX           7.0.0       ->  7.5.0  
    CUDA            10.1        ->  11.7
    nvcc c++        c++03,11,14 ->  c++03,11,14,17
    gcc             8.3.0       ->  11.2 
    NVIDIA Driver   435.21      ->  515+



OPTICKS_CUDA_NVCC_DIALECT envvar
------------------------------------
   
I briefly looked into CMake CUDA version detection within sysrap/CMakeLists.txt
but that requires switching on “LANGUAGES CUDA” on project lines and runs into 
complications with Clang detection in cmake/Modules/OpticksCUDAFlags.cmake
so I am not pursuing that approach as its too big of a change for the benefit anyhow.

https://bitbucket.org/simoncblyth/opticks/commits/80a37832613f24dd5413282e0603788a3734cde1

Instead in the above commit I add OPTICKS_CUDA_NVCC_DIALECT envvar 
sensitivity that you can try.   Note its a bit of a kludge as you
will need to touch CMakeLists.txt as CMake doesn’t notice changes to 
envvars. Also you will need to om-conf and rebuild everything 
when you want to change it. 



Observation with CUDA 11.8, gcc 11.3.1 from Lorenzo Capriotti
----------------------------------------------------------------

* gcc 11.3.1, CUDA 11.8 : build has lots of Thrust C++11 deprecation warnings 


::

    > I am using the following versions:
    > - gcc (GCC) 11.3.1 20221121 (Red Hat 11.3.1-4)
    > - Cuda compilation tools, release 11.8, V11.8.89 - Build cuda_11.8.r11.8/compiler.31833905_0
    > 
    > 
    > Please find here also the full list of warnings I mentioned, this time also when making sysrap:
    > 
    > === om-make-one : sysrap          /home/lcapriotti/opticks/sysrap                              /home/lcapriotti/optickslib/build/sysrap                     
    > [  1%] Building NVCC (Device) object CMakeFiles/SysRap.dir/SysRap_generated_SU.cu.o
    > [  1%] Built target PythonPH
    > [  1%] Built target PythonJSON
    > [  1%] Built target PythonGS
    > In file included from /usr/local/cuda-11.8/include/thrust/detail/config/config.h:27,
    >                  from /usr/local/cuda-11.8/include/thrust/detail/config.h:23,
    >                  from /usr/local/cuda-11.8/include/thrust/device_ptr.h:24,
    >                  from /home/lcapriotti/opticks/sysrap/SU.cu:7:
    > /usr/local/cuda-11.8/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
    >   131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
    >       |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                   
    > In file included from /usr/local/cuda-11.8/include/cub/util_arch.cuh:36,
    >                  from /usr/local/cuda-11.8/include/cub/detail/device_synchronize.cuh:20,
    >                  from /usr/local/cuda-11.8/include/thrust/system/cuda/detail/util.h:36,
    >                  from /usr/local/cuda-11.8/include/thrust/system/cuda/detail/malloc_and_free.h:29,
    >                  from /usr/local/cuda-11.8/include/thrust/system/detail/adl/malloc_and_free.h:42,
    >                  from /usr/local/cuda-11.8/include/thrust/system/detail/generic/memory.inl:20,
    >                  from /usr/local/cuda-11.8/include/thrust/system/detail/generic/memory.h:69,
    >                  from /usr/local/cuda-11.8/include/thrust/detail/reference.h:23,
    >                  from /usr/local/cuda-11.8/include/thrust/memory.h:25,
    >                  from /usr/local/cuda-11.8/include/thrust/device_ptr.h:25,
    >                  from /home/lcapriotti/opticks/sysrap/SU.cu:7:
    > /usr/local/cuda-11.8/include/cub/util_cpp_dialect.cuh:142:13: warning: CUB requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define CUB_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
    >   142 |      CUB_COMPILER_DEPRECATION_SOFT(C++14, C++11);



