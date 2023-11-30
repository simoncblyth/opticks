reference_versions
===================

The reference NVIDIA OptiX version used by Opticks is still 7.0.0 
which was built with CUDA 10.1 and 

Using other version sets will very likely have problems, please
report them to the mailing list. 

Opticks adopts the policy of first picking the OptiX version, 
then follows the release notes for that OptiX version to 
pick the CUDA version (using the exact version that was used to 
build the OptiX version) and the minimum NVIDIA Driver version 
to use.  

+-----------------+----------------+-------------------+--------------+
|                 |  Old Reference | Current Reference |  NEXT        |
+=================+================+===================+==============+
| NVIDIA OptiX    |  7.0.0         |    7.5.0          |  7.6.0?      | 
+-----------------+----------------+-------------------+--------------+
| NVIDIA Driver   |  435.21        |    515.43.04      |   525.89.02  |
+-----------------+----------------+-------------------+--------------+
| NVIDIA CUDA     |  10.1          |    11.7           |   12.0       |
+-----------------+----------------+-------------------+--------------+
| gcc             |  8.3.0         |    11.2.0         |   11.2.0     |
+-----------------+----------------+-------------------+--------------+

* NEXT: 525.89.02 is old driver for use with CUDA 12.0 ? Perhaps can update to allow 8.0.0 test



CUDA 10.1 : supported C++ dialects
-------------------------------------

* nvcc in CUDA 10.1 only supports c++03|c++11|c++14

* https://docs.nvidia.com/cuda/archive/10.1/cuda-compiler-driver-nvcc/index.html

4.2.3.11. --std {c++03|c++11|c++14} (-std)

Select a particular C++ dialect.
Allowed Values::

    c++03
    c++11
    c++14

Default

The default C++ dialect depends on the host compiler. 
nvcc matches the default C++ dialect that the host compiler uses.


CUDA and gcc version constraints
------------------------------------

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



Future Version Sets : Gleaned from NVIDIA OptiX Release Notes
---------------------------------------------------------------

+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Release Date    |   NVIDIA OptiX    |  Notes          |  Driver        |  CUDA   |  gcc    |                                |   
+==================+===================+=================+================+=========+=========+================================+
|  July 2019       |   :r:`7.0.0`      | **NEW API**     | 435.12(435.21) |  10.1   |  8.3.0  | <=current workstation versions |
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  June 2020       |   7.1.0           | Added Curves    | 450            |  11.0   |         | Maybe: torus for guide tube    |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2020        |   7.2.0           | Specialization  | 455            |  11.1   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Apr 2021        |   7.3.0           |                 | 465            |  11.1   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2021        |   7.4.0           | Catmull-Rom     | 495            |  11.4   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  June 2022       | :b:`7.5.0` [1]    | Debug, Sphere   | 515            |  11.7   |         | :b:`LOOKS POSSIBLE ON CLUSTER` |
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Oct 2022        |   7.6.0 [1]       |                 | 520            |  11.8   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Mar 2023        |   7.7.0           | More Curves     | 530            |  12.0   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+
|  Aug 2023        |   8.0.0           | SER, Perf       | 535            |  12.0   |         |                                |   
+------------------+-------------------+-----------------+----------------+---------+---------+--------------------------------+


* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
* https://docs.nvidia.com/cuda/archive/11.8.0/
* https://gist.github.com/ax3l/9489132


Fixed NVIDIA Driver + CUDA version => Fixed OptiX version
----------------------------------------------------------

For example when nvidia-smi gives::

    Thu Nov  2 15:33:23 2023    
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 515.65.01    Driver Version: 515.65.01    CUDA Version: 11.7     |   
    |-------------------------------+----------------------+----------------------+
    ...

Then the appropriate OptiX version is 7.5.0 



