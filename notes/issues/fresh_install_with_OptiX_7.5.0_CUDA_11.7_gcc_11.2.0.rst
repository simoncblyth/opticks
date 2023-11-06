fresh_install_with_OptiX_7.5.0_CUDA_11.7_gcc_11.2.0
=====================================================

Overview
----------

This testing uses the pure opticks install "ssh R" workstation/simon 
to avoid complications from JUNOSW. 

* 1st attempt with cvmfs gcc 11.2.0 fails in U4 with G4Exception methods undefined
* revert to cvmfs gcc 8.3.0 gave same issue
* revert to /devtoolset-8/enable    ## gcc 8.3.1

  * PRESUMABLY THIS WORKS BECAUSE ITS THE COMPILER USED FOR Geant4  


OKConf finds correct versions after PATH fix::

    -- OKCONF_OPTIX_INSTALL_DIR : 
    -- OptiX_VERSION_INTEGER : 70500
    -- OpticksCUDA_API_VERSION : 11070
    -- G4_VERSION_INTEGER      : 1042
    -- Configuring OKConfTest



DONE : build against Boost/Xercesc/CLHEP/Geant4 from /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830 
----------------------------------------------------------------------------------------------------

Works but its gcc830::

    opticks-prepend-prefix /data/blyth/junotop/ExternalLibs/Boost/1.78.0
    opticks-prepend-prefix /data/blyth/junotop/ExternalLibs/Xercesc/3.2.2
    opticks-prepend-prefix /data/blyth/junotop/ExternalLibs/CLHEP/2.4.1.0
    opticks-prepend-prefix /data/blyth/junotop/ExternalLibs/Geant4/10.04.p02.juno
        
Try with gcc1120::

   /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Boost/1.78.0
   /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Xercesc/3.2.3
   /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/CLHEP/2.4.1.0
   /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/Pre-Release/J23.1.x/ExternalLibs/Geant4/10.04.p02.juno




Issue 1 : C++11 warnings from thrust headers
----------------------------------------------

SU.cc and QEvent.cc C++11 warnings from thrust headers::

    [2023-11-06 14:45:54,791] p312626 {/home/simon/opticks/ana/enum_.py:151} INFO - writing ini to inipath /data/simon/local/opticks/build/sysrap/OpticksPhoton_Enum.ini 
    [  1%] Built target PythonGS
    [  1%] Built target PythonJSON
    [  1%] Built target PythonPH
    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/device_ptr.h:24,
                     from /home/simon/opticks/sysrap/SU.cu:7:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                                                                           

    [ 18%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QCerenkov.cu.o
    In file included from /usr/local/cuda-11.7/include/thrust/detail/config/config.h:27,
                     from /usr/local/cuda-11.7/include/thrust/detail/config.h:23,
                     from /usr/local/cuda-11.7/include/thrust/device_vector.h:25,
                     from /data/simon/local/opticks/include/SysRap/iexpand.h:60,
                     from /home/simon/opticks/qudarap/QEvent.cu:9:
    /usr/local/cuda-11.7/include/thrust/detail/config/cpp_dialect.h:131:13: warning: Thrust requires at least C++14. C++11 is deprecated but still supported. C++11 support will be removed in a future release. Define THRUST_IGNORE_DEPRECATED_CPP_DIALECT to suppress this message.
      131 |      THRUST_COMPILER_DEPRECATION_SOFT(C++14, C++11);
          |             ^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~                                                                                                                                                           

OpticksCUDAFlags.cmake is pinning the dialect to c++11::

    set(OPTICKS_CUDA_NVCC_DIALECT $ENV{OPTICKS_CUDA_NVCC_DIALECT})
    if(OPTICKS_CUDA_NVCC_DIALECT)
        message(STATUS "cmake/Modules/OpticksCUDAFlags.cmake : reading envvar OPTICKS_CUDA_NVCC_DIALECT into variable ${OPTICKS_CUDA_NVCC_DIALECT}")
    else()
        set(OPTICKS_CUDA_NVCC_DIALECT "c++11")
        message(STATUS "cmake/Modules/OpticksCUDAFlags.cmake : using default OPTICKS_CUDA_NVCC_DIALECT variable ${OPTICKS_CUDA_NVCC_DIALECT}")
    endif()
        

OpticksCXXFlags.cmake::

     69 
     70   if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
     71      set(CMAKE_CXX_STANDARD 14)
     72      set(CMAKE_CXX_STANDARD_REQUIRED on)
     73   else ()
     74      #set(CMAKE_CXX_STANDARD 14)
     75      set(CMAKE_CXX_STANDARD 17)   ## Geant4 1100 forcing c++17 : BUT that restricts to gcc 5+ requiring 
     76      set(CMAKE_CXX_STANDARD_REQUIRED on)
     77   endif ()


Issue 2 : Geant4 undefined references : FIXED BY USING SAME COMPILER FOR GEANT4 AND OPTICKS
----------------------------------------------------------------------------------------------

U4::

    [ 70%] Linking CXX executable U4TouchableTest
    [ 71%] Linking CXX executable U4NistManagerTest
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libU4.so: undefined reference to `G4Exception(char const*, char const*, G4ExceptionSeverity, std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >&, char const*)'
    /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bin/ld: ../libU4.so: undefined reference to `G4Exception(char const*, char const*, G4ExceptionSeverity, std::__cxx11::basic_ostringstream<char, std::char_traits<char>, std::allocator<char> >&)'
    collect2: error: ld returned 1 exit status
    make[2]: *** [tests/U4Hit_Debug_Test] Error 1
    make[1]: *** [tests/CMakeFiles/U4Hit_Debug_Test.dir/all] Error 2
    make[1]: *** Waiting for unfinished jobs....


The above was when trying to use cvmfs compiler to compile opticks but devtoolset
to compile geant4. 

Hmm : probably must reinstall geant4 with the same compiler. Instead of doing that 
return to the compiler used to build the geant4 installs and try fresh again. 



Try fresh_install with gcc 8.3.0 : using /opt/rh/devtoolset-8/enable that was used for geant4
-----------------------------------------------------------------------------------------------

That works with 2 ctest fails, so that means the devtoolset-8 gcc 8.3.1 is not compatible with the cvmfs 8.3.0 

~simon/.bashrc::

    # default gcc is 4.8.5 
    #source /opt/rh/devtoolset-7/enable    ## gcc 7.3.1 
    source /opt/rh/devtoolset-8/enable    ## gcc 8.3.1 
    #source /opt/rh/devtoolset-9/enable    ## gcc 9.3.1 : cannot be used with CUDA 10.1

    # follow lint example to use JUNO gcc830 
    # /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/Pre-Release/J21v2r0-Pre0/quick-deploy-J21v2r0-Pre0.sh
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/gcc/8.3.0/bashrc
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc830/contrib/binutils/2.28/bashrc

    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/gcc/11.2.0/bashrc
    #source /cvmfs/juno.ihep.ac.cn/centos7_amd64_gcc1120/contrib/binutils/2.36/bashrc


Near success::

    SLOW: tests taking longer that 15 seconds
      21 /32  Test #21 : U4Test.U4TreeTest                             Passed                         20.95  
      22 /32  Test #22 : U4Test.U4TreeCreateTest                       Passed                         21.08  
      23 /32  Test #23 : U4Test.U4TreeCreateSSimTest                   Passed                         21.10  


    FAILS:  2   / 208   :  Mon Nov  6 15:28:02 2023   
      76 /104 Test #76 : SysRapTest.stranTest                          ***Exception: Interrupt        0.01   
      11 /32  Test #11 : U4Test.U4RandomTest                           ***Failed                      0.06   


::

    sy ; CTESTARG="-R stranTest" om-test
    u4 ; CTESTARG="-R U4RandomTest" om-test

    NP::load Failed to load from path /home/simon/.opticks/InputPhotons/RandomDisc100_f8.npy
    NP::load Failed to load from path /home/simon/.opticks/precooked/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy



No fails after plant the symbolic links to access precooked and InputPhotons
-------------------------------------------------------------------------------

::

    (base) [simon@localhost .opticks]$ ln -s /home/blyth/.opticks/precooked
    (base) [simon@localhost .opticks]$ ln -s /home/blyth/.opticks/InputPhotons

    SLOW: tests taking longer that 15 seconds
      21 /32  Test #21 : U4Test.U4TreeTest                             Passed                         21.21  
      22 /32  Test #22 : U4Test.U4TreeCreateTest                       Passed                         20.97  
      23 /32  Test #23 : U4Test.U4TreeCreateSSimTest                   Passed                         21.05  


    FAILS:  0   / 208   :  Mon Nov  6 15:43:42 2023   



TODO : separate executable for precooking QSimTest::rng_sequence 
------------------------------------------------------------------------

* HMM: precooked randoms and input photons are non-essential, they are for debugging 
* YES: but its simpler if every install has them 




gcc1120 CUDA 11.7 new warnings : just a typo
-----------------------------------------------

::

    === om-make-one : CSGOptiX        /home/simon/opticks/CSGOptiX                                 /data/simon/local/opticks/build/CSGOptiX                     
    [  5%] Building NVCC ptx file CSGOptiX_generated_Check.cu.ptx
    [  5%] Building NVCC ptx file CSGOptiX_generated_CSGOptiX7.cu.ptx
    /data/simon/local/opticks/include/QUDARap/qsim.h(2195): warning #181-D: argument is incompatible with corresponding format string conversion

    /data/simon/local/opticks/include/QUDARap/qsim.h(2195): warning #224-D: the format string requires additional arguments






