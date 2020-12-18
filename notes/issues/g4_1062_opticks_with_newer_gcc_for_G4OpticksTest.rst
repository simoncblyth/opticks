g4_1062_opticks_with_newer_gcc_for_G4OpticksTest
==================================================


G4OpticksTest needs at least 1062
------------------------------------

But still a problem with old Geant4 1042::

    ...

    [ 66%] Built target G4OpticksTest
    [ 67%] Linking CXX shared library libG4OpticksTestClassesDict.so
    /usr/bin/ld: anHCAllocator_G4MT_TLS_: TLS definition in /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so section .tbss mismatches non-TLS reference in CMakeFiles/G4OpticksTestClassesDict.dir/src/PhotonSD.cc.o
    /home/blyth/junotop/ExternalLibs/Geant4/10.04.p02/lib64/libG4digits_hits.so: error adding symbols: Bad value
    collect2: error: ld returned 1 exit status
    make[2]: *** [libG4OpticksTestClassesDict.so] Error 1
    make[1]: *** [CMakeFiles/G4OpticksTestClassesDict.dir/all] Error 2
    make: *** [all] Error 2


Clear up some space then::

    [blyth@localhost ~]$ g4-;g4--1062

Nope::

    [ 13%] Building CXX object source/geometry/CMakeFiles/G4geometry.dir/magneticfield/src/G4TrialsCounter.cc.o
    -- [download 13% complete]
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/magneticfield/src/G4FieldManagerStore.cc: In static member function ‘static void G4FieldManagerStore::DeRegister(G4FieldManager*)’:
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02/source/geometry/magneticfield/src/G4FieldManagerStore.cc:119:31: error: no matching function for call to ‘G4FieldManagerStore::erase(__gnu_cxx::__normal_iterator<G4FieldManager* const*, std::vector<G4FieldManager*> >&)’
             GetInstance()->erase(i);
                                   ^

* https://geant4-forum.web.cern.ch/t/error-when-making-geant4/1774

gcosmo::

    You must use a more recent gcc compiler to build Geant4 10.6.
    The minimum required is gcc-4.9.3 and you’re using gcc-4.8.2…i

See env-;centos- for notes on yum installation of devtoolset-9 which comes with gcc 9.3.1 


Built g4_1062 using newer gcc under blyth account
----------------------------------------------------
     
Uncomment line in ~/.bash_profile:: 

  4 
  5 source /opt/rh/devtoolset-9/enable   ## gcc 9.3.1
  6 


Succeeded to build after deleting the old failed build dir::

    [blyth@localhost geant4.10.06.p02.Debug.build]$ pwd
    /home/blyth/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build
    [blyth@localhost geant4.10.06.p02.Debug.build]$ cd /home/blyth/local/opticks_externals/
    [blyth@localhost opticks_externals]$ l
    total 0
    drwxrwxr-x. 2 blyth blyth  6 Dec 16 23:24 g4_1062
    drwxrwxr-x. 4 blyth blyth 97 Dec 16 23:24 g4_1062.build
    drwxrwxr-x. 6 blyth blyth 58 Oct  5 21:06 g4_1042
    drwxrwxr-x. 4 blyth blyth 97 Oct  5 21:03 g4_1042.build
    [blyth@localhost opticks_externals]$ rm -rf g4_1062.build
    [blyth@localhost opticks_externals]$ 


What is needed to test G4OpticksTest ?
------------------------------------------

Hmm to test this need:

0. enable newer gccc
1. install clhep,boost,xercesc,g4_1062
2. build Opticks against those
3. build G4OpticksTest against those

Decide that is too disruptive to do in blyth account so adopt simon account for this.


Enable gcc 9.3.1 within simon account
----------------------------------------

Use S for this::

     63 P(){ ssh P ; }                           ## with JUNOTOP    : juno/opticks dev 
     64 S(){ ssh simon@P ; }                     ## without JUNOTOP : joe user 
     65 O(){ TERM=${TERM}@opticks ssh P ; }      ## normal Opticks dev 


Find that with S sourcing the below works interactively, but not from .bashrc ?

* pilot error : the sourcing must be placed after absolute PATH setup


/opt/rh/devtoolset-9/enable::

    # General environment variables
    export PATH=/opt/rh/devtoolset-9/root/usr/bin${PATH:+:${PATH}}
    export MANPATH=/opt/rh/devtoolset-9/root/usr/share/man:${MANPATH}
    export INFOPATH=/opt/rh/devtoolset-9/root/usr/share/info${INFOPATH:+:${INFOPATH}}
    export PCP_DIR=/opt/rh/devtoolset-9/root
    # bz847911 workaround:
    # we need to evaluate rpm's installed run-time % { _libdir }, not rpmbuild time
    # or else /etc/ld.so.conf.d files?
    rpmlibdir=$(rpm --eval "%{_libdir}")
    # bz1017604: On 64-bit hosts, we should include also the 32-bit library path.
    if [ "$rpmlibdir" != "${rpmlibdir/lib64/}" ]; then
      rpmlibdir32=":/opt/rh/devtoolset-9/root${rpmlibdir/lib64/lib}"
    fi
    export LD_LIBRARY_PATH=/opt/rh/devtoolset-9/root$rpmlibdir$rpmlibdir32${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export LD_LIBRARY_PATH=/opt/rh/devtoolset-9/root$rpmlibdir$rpmlibdir32:/opt/rh/devtoolset-9/root$rpmlibdir/dyninst$rpmlibdir32/dyninst${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
    export PKG_CONFIG_PATH=/opt/rh/devtoolset-9/root/usr/lib64/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}

     
::

    export PATH=/opt/rh/devtoolset-9/root/usr/bin${PATH:+:${PATH}}
    #  ${PATH:+:${PATH}}  if PATH exists and is non null add  :${PATH} otherwise add nothing 

.bashrc::

     30 # NB when enabling or disabling devtoolset-9 to get gcc 9.3.1 instead of 4.8.5 
     31 # start a new session and exit the old sessions for clarity
     32 # NB must do this after any absolute PATH settings as it prefixes PATH and LD_LIBRARY_PATH
     33 source /opt/rh/devtoolset-9/enable   ## gcc 9.3.1 vs default 4.8.5


::

    [simon@localhost CLHEP.build]$ gcc --version
    gcc (GCC) 9.3.1 20200408 (Red Hat 9.3.1-2)
    Copyright (C) 2019 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


Install foreign externals
--------------------------

Using newer gcc build and install the foreign externals::

   clhep-
   clhep-info   # default prefix is /home/simon/local/opticks_externals/clhep
   clhep--

   xercesc-
   xercesc-info # default prefix is /home/simon/local/opticks_externals/xercesc
   xercesc--


Possible pc issue::

    opticks-pc-rename-kludge
    ---------------------------

       name      : xerces-c
       name2     : OpticksXercesC
       pcfiledir : /usr/lib64/pkgconfig
       path      : /usr/lib64/pkgconfig/xerces-c.pc 
       path2     : /usr/lib64/pkgconfig/OpticksXercesC.pc
       path3     : /home/simon/local/opticks/externals/lib/pkgconfig/OpticksXercesC.pc

    === opticks-pc-rename-kludge : NO write permission to path3 /home/simon/local/opticks/externals/lib/pkgconfig/OpticksXercesC.pc either


::

   boost-
   boost-info    # default prefix is /home/simon/local/opticks_externals/boost
   boost--


::

   g4-
   OPTICKS_GEANT4_VER=1062 g4-info    # prefix is /home/simon/local/opticks_externals/g4_1062
   g4--1062

Oops forgot to add the prefix::

    -- Detecting CXX compile features - done
    CMake Error at cmake/Modules/Geant4OptionalComponents.cmake:64 (find_package):
      Could not find a package configuration file provided by "CLHEP" (requested
      version 2.3.3.0) with any of the following names:

        CLHEPConfig.cmake
        clhep-config.cmake

      Add the installation prefix of "CLHEP" to CMAKE_PREFIX_PATH or set
      "CLHEP_DIR" to a directory containing one of the above files.  If "CLHEP"
      provides a separate development package or SDK, be sure it has been
      installed.
    Call Stack (most recent call first):
      cmake/Modules/G4CMakeMain.cmake:59 (include)
      CMakeLists.txt:50 (include)


Not yet existing dirs just give warnings::

     28 ## hookup paths to access "foreign" externals 
     29 ext=/home/simon/local/opticks_externals
     30 opticks-prepend-prefix $ext/boost
     31 opticks-prepend-prefix $ext/clhep
     32 opticks-prepend-prefix $ext/xercesc
     33 opticks-prepend-prefix $ext/g4_1062 
     34 



opticks-full looking for 1040 ?::

    [simon@localhost ~]$ opticks-full

    ...

    ############## g4 ###############


    -bash: /home/simon/local/opticks_externals/g4_1042/bin/geant4-config: No such file or directory
    generate /home/simon/local/opticks_externals/g4_1042//pkgconfig/Geant4.pc
    -bash: /home/simon/local/opticks_externals/g4_1042/bin/geant4-config: No such file or directory
    -bash: /home/simon/local/opticks_externals/g4_1042/bin/geant4-config: No such file or directory
    -bash: /home/simon/local/opticks_externals/g4_1042/bin/geant4-config: No such file or directory
    === opticks-full-externals : DONE Sat Dec 19 00:21:02 CST 2020
    === opticks-full-make : START Sat Dec 19 00:21:02 CST 2020
    === opticks-full-make : generating setup script
    === opticks-check-geant4 : ERROR no g4_prefix : failed to find Geant4Config.cmake along CMAKE_PREFIX_PATH
    [simon@localhost nljson]$ 


::

    [simon@localhost ~]$ opticks-foreign
    boost
    clhep
    xercesc
    g4
    [simon@localhost ~]$ t opticks-foreign-pc
    opticks-foreign-pc () 
    { 
        opticks-pc- $(opticks-foreign)
    }
    [simon@localhost ~]$ t opticks-pc-
    opticks-pc- () 
    { 
        echo $FUNCNAME;
        local msg="=== $FUNCNAME :";
        local funcs=$*;
        local func;
        for func in $funcs;
        do
            printf "\n\n\n############## %s ###############\n\n\n" $func;
            $func-;
            $func-pc;
            rc=$?;
            [ $rc -ne 0 ] && echo $msg RC $rc from func $func : ABORTING && return $rc;
        done;
        return 0
    }
    [simon@localhost ~]$ 


Need to tell the opticks-full to use the different G4::



    g4--1062 () 
    { 
        OPTICKS_GEANT4_VER=1062 g4--
    }

    simon@localhost nljson]$ t g4-prefix
    g4-prefix () 
    { 
        echo ${OPTICKS_GEANT4_PREFIX:-$(opticks-prefix)_externals/g4_$(g4-ver)}
    }
    [simon@localhost nljson]$ g4-ver
    1042
    [simon@localhost nljson]$ t g4-ver
    g4-ver () 
    { 
        echo ${OPTICKS_GEANT4_VER:-1042}
    }
    [simon@localhost nljson]$ 

::

    [simon@localhost nljson]$ vi ~/.opticks_config  # add:  export OPTICKS_GEANT4_VER=1062
    [simon@localhost nljson]$ ini
    [simon@localhost nljson]$ g4-prefix
    /home/simon/local/opticks_externals/g4_1062

    [simon@localhost nljson]$ g4-pc
    generate /home/simon/local/opticks_externals/g4_1062/lib64/pkgconfig/Geant4.pc


Continue with opticks-full-make, runs in cuda problem::

    === om-make-one : cudarap         /home/simon/opticks/cudarap                                  /home/simon/local/opticks/build/cudarap                      
    [  4%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_cuRANDWrapper_kernel.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CResource_.cu.o
    [ 13%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CDevice.cu.o
    In file included from /usr/local/cuda-10.1/include/cuda_runtime.h:83,
                     from <command-line>:
    /usr/local/cuda-10.1/include/crt/host_config.h:129:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
      129 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!
          |  ^~~~~
    In file included from /usr/local/cuda-10.1/include/cuda_runtime.h:83,
                     from <command-line>:
    /usr/local/cuda-10.1/include/crt/host_config.h:129:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
      129 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!
          |  ^~~~~
    In file included from /usr/local/cuda-10.1/include/cuda_runtime.h:83,
                     from <command-line>:
    /usr/local/cuda-10.1/include/crt/host_config.h:129:2: error: #error -- unsupported GNU version! gcc versions later than 8 are not supported!
      129 | #error -- unsupported GNU version! gcc versions later than 8 are not supported!
          |  ^~~~~
    CMake Error at CUDARap_generated_CResource_.cu.o.Debug.cmake:219 (message):
      Error generating
      /home/simon/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//./CUDARap_generated_CResource_.cu.o



gcc 9 not supported with CUDA 10.1 
------------------------------------

* https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version

From OptiX 6.5 release notes. August 26, 2019
--------------------------------------------------

OptiX 6.5.0 has been built with CUDA 10.1, but any specified toolkit should work when compiling PTX for OptiX.
OptiX uses the CUDA device API, but the CUDA runtime API objects can be cast to device API objects.

C/C++ Compiler : A compiler compatible with the CUDA Toolkit version used is required. 
Please see the CUDA Toolkit documentation for more information on supported compilers.


CUDA Toolkit
-------------

* https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html

* https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-linux/index.html

::

    Distrib     kernel  GCC     GLIBC
    RHEL 8.0	4.18	8.2.1	2.28	 	 	 	 
    RHEL 7.6	3.10	4.8.5	2.17
    RHEL 6.10	2.6.32	4.4.7	2.12
    CentOS 7.6	3.10	4.8.5	2.17
    CentOS 6.10	2.6.32	4.4.7	2.12
    Fedora 29	4.16	8.0.1	2.27
    OpenSUSE Leap 15.0	4.15.0	7.3.1	2.26
    SLES 15.0	4.12.14	7.2.1	2.26
    SLES 12.4	4.12.14	4.8.5	2.22
    Ubuntu 18.10	4.18.0	8.2.0	2.28
    Ubuntu 18.04.3 (**)	5.0.0	7.4.0	2.27
    Ubuntu 16.04.6 (**)	4.4	5.4.0	2.23
    Ubuntu 14.04.6 (**)	3.13	4.8.4	2.19


devtoolset
------------

See what gcc version devtoolset-8 gives 

.bashrc::

    devtoolset-notes(){ cat << EON
    When enabling/disabling/changing devtoolset
    ---------------------------------------------

    1. start a new session and exit the old sessions for clarity
    2. must do this after any absolute PATH settings as it prefixes PATH and LD_LIBRARY_PATH

    * https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version

    EON
    }
    # default gcc is 4.8.5 
    #source /opt/rh/devtoolset-9/enable    ## gcc 9.3.1 cannot be used with CUDA 10
    source /opt/rh/devtoolset-8/enable     ## gcc 8.3.1 



Do something dirty try to resume the build with different compiler... no chance::


    [ 13%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_cuRANDWrapper_kernel.cu.o
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6725:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
     ^     ~~~~~~~~~
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic


::

   o
   cd examples/UseCUDA
   cd examples/UseUseCUDA
   ## these work with gcc 8.3.1



Delete everything from gcc 9
---------------------------------


::

    [simon@localhost ~]$ l /home/simon/local/
    total 0
    drwxrwxr-x. 10 simon simon 114 Dec 19 00:32 opticks
    drwxrwxr-x. 11 simon simon 157 Dec 19 00:21 opticks_externals
    [simon@localhost ~]$ l /home/simon/local/opticks_externals/
    total 0
    drwxrwxr-x. 4 simon simon 32 Dec 18 22:33 boost
    drwxrwxr-x. 4 simon simon 79 Dec 18 22:30 boost.build
    drwxrwxr-x. 5 simon simon 43 Dec 18 22:19 clhep
    drwxrwxr-x. 3 simon simon 46 Dec 18 22:12 clhep.build
    drwxrwxr-x. 3 simon simon 23 Dec 19 00:21 g4_1042
    drwxrwxr-x. 6 simon simon 58 Dec 18 23:59 g4_1062
    drwxrwxr-x. 4 simon simon 97 Dec 18 22:55 g4_1062.build
    drwxrwxr-x. 5 simon simon 43 Dec 18 22:26 xercesc
    drwxrwxr-x. 3 simon simon 57 Dec 18 22:21 xercesc.build
    [simon@localhost ~]$ 

    simon@localhost ~]$ du -hs /home/simon/local/*
    807M	/home/simon/local/opticks
    9.4G	/home/simon/local/opticks_externals

    [simon@localhost ~]$ rm -rf /home/simon/local
    [simon@localhost ~]$ mkdir -p /home/simon/local


Back to beginning with devtoolset-8  : opticks-foreign-install
----------------------------------------------------------------

::

    [simon@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

::

    [simon@localhost ~]$ echo $OPTICKS_GEANT4_VER
    1062

    [simon@localhost ~]$ g4-prefix    # thanks to the OPTICKS_GEANT4_VER envvar 
    /home/simon/local/opticks_externals/g4_1062


::

    [simon@localhost ~]$ opticks-
    [simon@localhost ~]$ opticks-foreign
    boost
    clhep
    xercesc
    g4
    [simon@localhost ~]$ opticks-foreign-install





After that : opticks-full 
-----------------------------




