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
       
     


With gcc 8.3.1 and cmake 3.17 the warning is appearing again for g4ok::


      6 if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
      7     cmake_policy(SET CMP0077 OLD)
      8 endif()
      9 


   


cmake/Modules/OpticksCUDAFlags.cmake::

     07 set(CUDA_NVCC_FLAGS)
      8 
      9 if(NOT (COMPUTE_CAPABILITY LESS 30))
     10 
     11    #list(APPEND CUDA_NVCC_FLAGS "-arch=sm_${COMPUTE_CAPABILITY}")
     12    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler -fPIC")
     13    list(APPEND CUDA_NVCC_FLAGS "-gencode=arch=compute_${COMPUTE_CAPABILITY},code=sm_${COMPUTE_CAPABILITY}")
     14 
     15    #list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
     16    # https://github.com/facebookresearch/Detectron/issues/185
     17 
     18    list(APPEND CUDA_NVCC_FLAGS "-O2")
     19    #list(APPEND CUDA_NVCC_FLAGS "-DVERBOSE")
     20    list(APPEND CUDA_NVCC_FLAGS "--use_fast_math")
     21 
     22    #list(APPEND CUDA_NVCC_FLAGS "-m64")
     23    #list(APPEND CUDA_NVCC_FLAGS "--disable-warnings")
     24 
     25    set(CUDA_PROPAGATE_HOST_FLAGS OFF)
     26    set(CUDA_VERBOSE_BUILD OFF)
     27 
     28 endif()
     29 
      


::

    [simon@localhost cfg4]$ cmake --help-policy CMP0077
    CMP0077
    -------

    ``option()`` honors normal variables.

    The ``option()`` command is typically used to create a cache entry
    to allow users to set the option.  However, there are cases in which a
    normal (non-cached) variable of the same name as the option may be
    defined by the project prior to calling the ``option()`` command.
    For example, a project that embeds another project as a subdirectory
    may want to hard-code options of the subproject to build the way it needs.

    For historical reasons in CMake 3.12 and below the ``option()``
    command *removes* a normal (non-cached) variable of the same name when:

    * a cache entry of the specified name does not exist at all, or
    * a cache entry of the specified name exists but has not been given
      a type (e.g. via ``-D<name>=ON`` on the command line).

    In both of these cases (typically on the first run in a new build tree),
    the ``option()`` command gives the cache entry type ``BOOL`` and
    removes any normal (non-cached) variable of the same name.  In the
    remaining case that the cache entry of the specified name already
    exists and has a type (typically on later runs in a build tree), the
    ``option()`` command changes nothing and any normal variable of
    the same name remains set.

    In CMake 3.13 and above the ``option()`` command prefers to
    do nothing when a normal variable of the given name already exists.
    It does not create or update a cache entry or remove the normal variable.
    The new behavior is consistent between the first and later runs in a
    build tree.  This policy provides compatibility with projects that have
    not been updated to expect the new behavior.

    When the ``option()`` command sees a normal variable of the given
    name:

    * The ``OLD`` behavior for this policy is to proceed even when a normal
      variable of the same name exists.  If the cache entry does not already
      exist and have a type then it is created and/or given a type and the
      normal variable is removed.

    * The ``NEW`` behavior for this policy is to do nothing when a normal
      variable of the same name exists.  The normal variable is not removed.
      The cache entry is not created or updated and is ignored if it exists.

    This policy was introduced in CMake version 3.13.  CMake version
    3.13.4 warns when the policy is not set and uses ``OLD`` behavior.
    Use the ``cmake_policy()`` command to set it to ``OLD`` or ``NEW``
    explicitly.

    .. note::
      The ``OLD`` behavior of a policy is
      ``deprecated by definition``
      and may be removed in a future version of CMake.
    [simon@localhost cfg4]$ 






g4ok::

    -- Looking for pthread_create in pthread - found
    -- Found Threads: TRUE  
    -- FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll Geant4::G4geometry;Geant4::G4global;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4materials;Geant4::G4particles;Geant4::G4digits_hits;Geant4::G4event;Geant4::G4processes;Geant4::G4run;Geant4::G4track;Geant4::G4tracking;XercesC::XercesC 
    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:576 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
    Call Stack (most recent call first):
      /home/simon/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cudarap/cudarap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/thrustrap/thrustrap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at /usr/share/cmake3/Modules/FindCUDA.cmake:582 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_VERBOSE_BUILD'.
    Call Stack (most recent call first):
      /home/simon/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cudarap/cudarap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/thrustrap/thrustrap-config.cmake:18 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /home/simon/local/opticks/lib64/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Found CUDA: /usr/local/cuda-10.1 (found version "10.1") 
    -- Configuring G4OKTest


/usr/share/cmake3/Modules/FindCUDA.cmake::

     574 
     575 # Propagate the host flags to the host compiler via -Xcompiler
     576 option(CUDA_PROPAGATE_HOST_FLAGS "Propagate C/CXX_FLAGS and friends to the host compiler via -Xcompile" ON)
     577 
     578 # Enable CUDA_SEPARABLE_COMPILATION
     579 option(CUDA_SEPARABLE_COMPILATION "Compile CUDA objects with separable compilation enabled.  Requires CUDA 5.0+" OFF)
     580 
     581 # Specifies whether the commands used when compiling the .cu file will be printed out.
     582 option(CUDA_VERBOSE_BUILD "Print out the commands run while compiling the CUDA source file.  With the Makefile generator this defaults to VERBOSE variable specified on the command line, but can      be forced on with this option." OFF)
     583 






g4ok om-cleaninstall gives the above warnings with cmake 3.13.4 but not with 3.17.0::

    [simon@localhost g4ok]$ cmake --version
    cmake3 version 3.13.4

    CMake suite maintained and supported by Kitware (kitware.com/cmake).
    [simon@localhost g4ok]$ which cmake
    /usr/bin/cmake
    [simon@localhost g4ok]$ 



::

    [blyth@localhost opticks]$ cmake --version
    cmake version 3.17.0

    CMake suite maintained and supported by Kitware (kitware.com/cmake).
    [blyth@localhost opticks]$ which cmake
    ~/junotop/ExternalLibs/Cmake/3.17.0/bin/cmake



with cmake 


    -- Configuring SysRap
    CMake Warning (dev) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v2r0-branch/ExternalLibs/Cmake/3.21.2/share/cmake-3.21/Modules/FindCUDA.cmake:723 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
    Call Stack (most recent call first):
      CMakeLists.txt:24 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v2r0-branch/ExternalLibs/Cmake/3.21.2/share/cmake-3.21/Modules/FindCUDA.cmake:729 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_VERBOSE_BUILD'.
    Call Stack (most recent call first):
      CMakeLists.txt:24 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Looking for pthread.h



CAUTION : Have switched to policy NEW so there is potential for changed CUDA flags 
-------------------------------------------------------------------------------------


