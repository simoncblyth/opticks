cmake-3.13.4-FindCUDA-warnings : Quelled 
==========================================

om-conf of all the marked subs giving warnings Policy CMP0077 warnings::


    [blyth@localhost build]$ om-subs
    okconf
    sysrap
    boostrap
    npy
    yoctoglrap
    optickscore
    ggeo
    assimprap
    openmeshrap
    opticksgeo
    *cudarap*
    *thrustrap*
    *optixrap*
    *okop*
    oglrap
    *opticksgl*
    *ok*
    extg4
    cfg4
    *okg4*
    *g4ok*

::

    -- Configuring CUDARap
    -- OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : 70
    -- OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : -Xcompiler -fPIC;-gencode=arch=compute_70,code=sm_70;-O2;--use_fast_math 
    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:576 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
    Call Stack (most recent call first):
      /home/blyth/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      CMakeLists.txt:10 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:582 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_VERBOSE_BUILD'.
    Call Stack (most recent call first):
      /home/blyth/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      CMakeLists.txt:10 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Looking for pthread.h
    -- Looking for pthread.h - found



Quelled this warning with the policy. An attempt to do this once for all subs 
in OpticksBuildOpticks didnt work::

      1 cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
      2 set(name CUDARap)
      3 project(${name} VERSION 0.1.0)
      4 include(OpticksBuildOptions)
      5 
      6 if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
      7     cmake_policy(SET CMP0077 OLD)
      8 endif()
      9 




/usr/share/cmake3/Modules/FindCUDA.cmake::

     575 # Propagate the host flags to the host compiler via -Xcompiler
     576 option(CUDA_PROPAGATE_HOST_FLAGS "Propagate C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)
     577 
     578 # Enable CUDA_SEPARABLE_COMPILATION
     579 option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+" OFF)
     580 
     581 # Specifies whether the commands used when compiling the .cu file will be printed out.
     582 option(CUDA_VERBOSE_BUILD "Print out the commands run while compiling the CUDA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can be forced on with this option." OFF)
     583 
       
        

