g4_1062_opticks_with_newer_gcc_for_G4OpticksTest
==================================================


.. contents:: Table of Contents : Linux 
   :depth: 3 


See Also 
----------

* :doc:`g4_1062_opticks_with_newer_gcc_for_G4OpticksTest_Darwin`


G4OpticksTest needs at least Geant4 1062
--------------------------------------------

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

Decide that is too disruptive to do in blyth account so adopt Linux.simon and Darwin.charles accounts for this.



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

    Distrib             kernel  GCC     GLIBC
    RHEL 8.0	        4.18	8.2.1	2.28	 	 	 	 
    RHEL 7.6	        3.10	4.8.5	2.17
    RHEL 6.10	        2.6.32	4.4.7	2.12
    CentOS 7.6     	    3.10	4.8.5	2.17
    CentOS 6.10	        2.6.32	4.4.7	2.12
    Fedora 29	        4.16	8.0.1	2.27
    OpenSUSE Leap 15.0	4.15.0	7.3.1	2.26
    SLES 15.0	        4.12.14	7.2.1	2.26
    SLES 12.4	        4.12.14	4.8.5	2.22
    Ubuntu 18.10	    4.18.0	8.2.0	2.28
    Ubuntu 18.04.3 (**)	5.0.0	7.4.0	2.27
    Ubuntu 16.04.6 (**)	4.4	    5.4.0	2.23
    Ubuntu 14.04.6 (**)	3.13	4.8.4	2.19


::

    [simon@localhost ~]$ uname -a
    Linux localhost.localdomain 3.10.0-957.10.1.el7.x86_64 #1 SMP Mon Mar 18 15:06:45 UTC 2019 x86_64 x86_64 x86_64 GNU/Linux

    [simon@localhost ~]$ cat /etc/centos-release
    CentOS Linux release 7.6.1810 (Core) 

    [blyth@localhost ~]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.


It looks like CUDA might pin you to the standard gcc version for your kernel.
But plough on regardless to see what error you get.


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



After that : opticks-full  : runs into cudarap issue
-------------------------------------------------------

::

    === om-make-one : cudarap         /home/simon/opticks/cudarap                                  /home/simon/local/opticks/build/cudarap                      
    [  4%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CDevice.cu.o
    [  8%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CResource_.cu.o
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
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6730:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’ without object
    CMake Error at CUDARap_generated_CResource_.cu.o.Debug.cmake:279 (message):
      Error generating file
      /home/simon/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//./CUDARap_generated_CResource_.cu.o


    make[2]: *** [CMakeFiles/CUDARap.dir/CUDARap_generated_CResource_.cu.o] Error 1
    make[2]: *** Waiting for unfinished jobs....
    /home/simon/opticks/cudarap/CDevice.cu: In static member function ‘static void CDevice::Collect(std::vector<CDevice>&, bool)’:
    /home/simon/opticks/cudarap/CDevice.cu:71:25: warning: argument to ‘sizeof’ in ‘char* strncpy(char*, const char*, size_t)’ call is the same expression as the source; did you mean to use the size of the destination? [-Wsizeof-pointer-memaccess]
             strncpy( d.name, p.name, sizeof(p.name) );
                             ^~~~~~~~~~~~~~~
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6725:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
     ^     ~~~~~~~~~
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6730:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’ without object
    CMake Error at CUDARap_generated_CDevice.cu.o.Debug.cmake:279 (message):
      Error generating file
      /home/simon/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//./CUDARap_generated_CDevice.cu.o

    ...

    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /home/simon/local/opticks/build/cudarap : non-zero rc 2
    === opticks-prepare-installation : generating RNG seeds into installcache


* https://github.com/pytorch/vision/issues/1893

::

    [simon@localhost ~]$ which nvcc
    /usr/local/cuda-10.1/bin/nvcc
    [simon@localhost ~]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Fri_Feb__8_19:08:17_PST_2019
    Cuda compilation tools, release 10.1, V10.1.105
    [simon@localhost ~]$ 


* https://forums.developer.nvidia.com/t/cuda-10-1-nvidia-youre-now-fixing-gcc-bugs-that-gcc-doesnt-even-have/71063/5

::

    yep. but this depend in what system is installed. if you see in the system requeriments:

    Ubuntu 18.10 supports gcc 8.2.0
    Fedora 29 supports gcc 8.0.2

    thats why nvcc supports gcc 8 series. but you only can use it in that distros, or if update the glib/gcc to the same versions, then you can use it

    for this in arch use gcc 7 (7.4.1) for build cuda code instead of default gcc (8.2.1)

    see for example this issue in the incubator-mxnet project:


CUDA devtoolset-8 ?
---------------------

* :google:`CUDA devtoolset` 

* https://forums.developer.nvidia.com/t/rhel-centos-7-5-with-devtoolset-7-gcc-v-7-3-1-and-cuda-toolkit-v-10-0-130-compile-issue/68004

* https://stackoverflow.com/questions/60817809/using-cuda-thrust-in-existing-c-project-compilation-error


When I compile the code from godbolt on RHEL7, CUDA 10.1.243, gcc 4.8.5, it
compiles cleanly for me. The last gcc 8 version that was tested with CUDA
10.1.243 is 8.2.1, not 8.3, so its possible there is a difference there. But if
you say that you switched to gcc 4.8.5 and it didn't fix anything, then I'm
quite confident your host environment is messed up. Those claims are not all
supportable. If you switched to gcc 4.8.5, and you are still getting errors of
the form /opt/rh/devtoolset-8/..., then my claim is you did not switch to
gcc4.8.5 (correctly). – Robert Crovella Mar 23 at 17:28

::

    [simon@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.




devtoolset-7 ?
-----------------

Get similar errors with 


::

    [simon@localhost tests]$ pwd
    /home/simon/opticks/thrustrap/tests
    [simon@localhost tests]$ 


    [simon@localhost tests]$ nvcc rng.cu -o /tmp/rng
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5033:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5054:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:6676:95:   required from here
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:1067:16: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
           ~~~~~~~~~^~
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5033:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:5054:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.h:6681:95:   required from here
    /opt/rh/devtoolset-7/root/usr/include/c++/7/bits/basic_string.tcc:1067:16: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’ without object




Observe that only the standard gcc works with CUDA nvcc : seems devtoolset changing gcc version doesnt work for CUDA
---------------------------------------------------------------------------------------------------------------------

::

    epsilon:opticks blyth$ S
    Warning: Permanently added '[127.0.0.1]:2001' (ECDSA) to the list of known hosts.
    Last login: Sat Dec 19 03:14:53 2020 from lxslc706.ihep.ac.cn
    [simon@localhost ~]$ cd /home/simon/opticks/thrustrap/tests
    [simon@localhost tests]$ gcc --version
    gcc (GCC) 4.8.5 20150623 (Red Hat 4.8.5-39)
    Copyright (C) 2015 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    [simon@localhost tests]$ nvcc rng.cu -o /tmp/rng
    [simon@localhost tests]$ /tmp/rng
          0 :    0.740219   0.438451   0.517013   0.156989   0.071368   0.462508   0.227643   0.329358   0.144065   0.187799   0.915383   0.540125   0.974661   0.547469   0.653160   0.230238
          1 :    0.920994   0.460364   0.333464   0.372520   0.489602   0.567271   0.079906   0.233368   0.509378   0.088979   0.006710   0.954227   0.546711   0.824547   0.527063   0.930132
          2 :    0.039020   0.250215   0.184484   0.962422   0.520555   0.939965   0.830578   0.409733   0.081622   0.806771   0.695286   0.617707   0.256335   0.213682   0.342424   0.224079
          3 :    0.968963   0.494743   0.673381   0.562773   0.120194   0.976486   0.135831   0.588972   0.490618   0.328445   0.911430   0.190679   0.963701   0.897554   0.624288   0.710151
          4 :    0.925141   0.053011   0.163102   0.889695   0.566639   0.241424   0.493690   0.321228   0.078608   0.147878   0.598657   0.426472   0.243465   0.489182   0.409532   0.667640



* :google:`devtoolset changing gcc version doesnt work for CUDA`


* https://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version

This isn't s binary compatibility question. The CUDA toolchain requires that
nvcc and the GPU front end parser can intercept and overload various compiler
and libc/libc++ internal headers to both compile host and device code and
integrate them together. The CUDA parser needs to be able to parse the gcc
internal headers correctly, amongst other things. Untested gcc versions can and
do fail, irrespective of preprocessor guards built into the NVIDIA headers. You
can either believe me (as someone who has been hacking on the CUDA toolchain
for almost 10 years), or not. At this point I don't really – talonmies Nov 28
'16 at 20:33



Resume the cudarap : it looks like adding the -std=c++11 works with centos7 + devtoolset-8 gcc 8.3.1 + cuda 10.1 
-------------------------------------------------------------------------------------------------------------------------

::

    [simon@localhost ~]$ gcc --version
    gcc (GCC) 8.3.1 20190311 (Red Hat 8.3.1-3)
    Copyright (C) 2018 Free Software Foundation, Inc.
    This is free software; see the source for copying conditions.  There is NO
    warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

    [simon@localhost ~]$ nvcc --version
    nvcc: NVIDIA (R) Cuda compiler driver
    Copyright (c) 2005-2019 NVIDIA Corporation
    Built on Fri_Feb__8_19:08:17_PST_2019
    Cuda compilation tools, release 10.1, V10.1.105
    [simon@localhost ~]$ 


Rapidly run into same error::

      
    [simon@localhost ~]$ sysrap
    [simon@localhost sysrap]$ om

    ...

    [simon@localhost sysrap]$ cudarap
    [simon@localhost cudarap]$ om
    === om-env : normal running
    ...


    [ 13%] Building NVCC (Device) object CMakeFiles/CUDARap.dir/CUDARap_generated_CDevice.cu.o
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6725:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
     ^     ~~~~~~~~~
     



Switch to VERBOSE to grab the compilation commandline that is failing::

    [simon@localhost cudarap]$ export VERBOSE=1
    [simon@localhost cudarap]$ om
    ...


    /usr/local/cuda-10.1/bin/nvcc /home/simon/opticks/cudarap/cuRANDWrapper_kernel.cu -c -o /home/simon/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//./CUDARap_generated_cuRANDWrapper_kernel.cu.o -ccbin /opt/rh/devtoolset-8/root/usr/bin/cc -m64 -DCUDARap_EXPORTS -DOPTICKS_CUDARAP -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -Xcompiler ,\"-fPIC\" -Xcompiler -fPIC -gencode=arch=compute_70,code=sm_70 -O2 --use_fast_math -DNVCC -I/usr/local/cuda-10.1/include -I/home/simon/opticks/cudarap -I/home/simon/local/opticks/include/SysRap -I/home/simon/local/opticks/externals/plog/include -I/home/simon/local/opticks/include/OKConf -I/usr/local/cuda-10.1/samples/common/inc



Paste the commandline into t.sh::

    #!/bin/bash -l

    notes(){ cat << EON
    notes/issues/g4_1062_opticks_with_newer_gcc_for_G4OpticksTest.rst
    EON
    }


    /usr/local/cuda-10.1/bin/nvcc /home/simon/opticks/cudarap/cuRANDWrapper_kernel.cu -c \
        -o /home/simon/local/opticks/build/cudarap/CMakeFiles/CUDARap.dir//./CUDARap_generated_cuRANDWrapper_kernel.cu.o \
        -ccbin /opt/rh/devtoolset-8/root/usr/bin/cc -m64 \
           -std=c++11 \
          -DCUDARap_EXPORTS -DOPTICKS_CUDARAP -DOPTICKS_SYSRAP -DOPTICKS_OKCONF -Xcompiler ,\"-fPIC\" -Xcompiler -fPIC \
         -gencode=arch=compute_70,code=sm_70 -O2 --use_fast_math -DNVCC \
           -I/usr/local/cuda-10.1/include \
           -I/home/simon/opticks/cudarap \
           -I/home/simon/local/opticks/include/SysRap \
           -I/home/simon/local/opticks/externals/plog/include \
           -I/home/simon/local/opticks/include/OKConf \
           -I/usr/local/cuda-10.1/samples/common/inc


Observe that t.sh duplicates the fail and adding "-std=c++11" fixes it. 
So the issue is that nvcc with gcc 8.3.1 requires that option whereas nvcc with the default gcc 4.8.5 does not.

::

    [simon@localhost ~]$ ./t.sh 
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char16_t*; _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6725:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char16_t; _Traits = std::char_traits<char16_t>; _Alloc = std::allocator<char16_t>]’ without object
           __p->_M_set_sharable();
     ^     ~~~~~~~~~
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc: In instantiation of ‘static std::basic_string<_CharT, _Traits, _Alloc>::_Rep* std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_S_create(std::basic_string<_CharT, _Traits, _Alloc>::size_type, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’:
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:578:28:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&, std::forward_iterator_tag) [with _FwdIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5052:20:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct_aux(_InIterator, _InIterator, const _Alloc&, std::__false_type) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:5073:24:   required from ‘static _CharT* std::basic_string<_CharT, _Traits, _Alloc>::_S_construct(_InIterator, _InIterator, const _Alloc&) [with _InIterator = const char32_t*; _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:656:134:   required from ‘std::basic_string<_CharT, _Traits, _Alloc>::basic_string(const _CharT*, std::basic_string<_CharT, _Traits, _Alloc>::size_type, const _Alloc&) [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>; std::basic_string<_CharT, _Traits, _Alloc>::size_type = long unsigned int]’
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.h:6730:95:   required from here
    /opt/rh/devtoolset-8/root/usr/include/c++/8/bits/basic_string.tcc:1067:1: error: cannot call member function ‘void std::basic_string<_CharT, _Traits, _Alloc>::_Rep::_M_set_sharable() [with _CharT = char32_t; _Traits = std::char_traits<char32_t>; _Alloc = std::allocator<char32_t>]’ without object
    [simon@localhost ~]$ vi t.sh 
    [simon@localhost ~]$ vi t.sh 
    [simon@localhost ~]$ ./t.sh 



cmake/Modules/OpticksCUDAFlags.cmake hmm it is there but commented::

     08 
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
     30 
     31 if(FLAGS_VERBOSE)
     32    message(STATUS "OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : ${COMPUTE_CAPABILITY}")
     33    message(STATUS "OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : ${CUDA_NVCC_FLAGS} ")
     34 endif()
     35 
     36 


::

    [simon@localhost opticks]$ cudarap
    [simon@localhost cudarap]$ om-conf
    bash: om-conf: command not found...
    [simon@localhost cudarap]$ om-
    === om-env : normal running
    === opticks-setup.sh : build time OPTICKS_PREFIX /home/simon/local/opticks is consistent with HERE_OPTICKS_PREFIX /home/simon/local/opticks
    === opticks-setup.sh : WARNING inconsistent CMAKE_PREFIX_PATH between build time and setup time
    === opticks-setup.sh : CMAKE_PREFIX_PATH         
    /home/simon/local/opticks_externals/g4_1062
    /home/simon/local/opticks_externals/xercesc
    /home/simon/local/opticks_externals/clhep
    /home/simon/local/opticks_externals/boost
    === opticks-setup.sh : BUILD_CMAKE_PREFIX_PATH   
    /home/simon/local/opticks_externals/g4_1062
    /home/simon/local/opticks_externals/xercesc
    /home/simon/local/opticks_externals/clhep
    /home/simon/local/opticks_externals/boost
    /home/simon/local/opticks
    /home/simon/local/opticks/externals
    /home/blyth/local/opticks/externals/OptiX_650

    === opticks-setup.sh : WARNING inconsistent PKG_CONFIG_PATH between build time and setup time
    === opticks-setup.sh : PKG_CONFIG_PATH           
    /home/simon/local/opticks_externals/g4_1062/lib64/pkgconfig
    /home/simon/local/opticks_externals/xercesc/lib/pkgconfig
    /home/simon/local/opticks_externals/clhep/lib/pkgconfig
    /home/simon/local/opticks_externals/boost/lib/pkgconfig
    === opticks-setup.sh : BUILD_PKG_CONFIG_PATH     
    /home/simon/local/opticks_externals/g4_1062/lib64/pkgconfig
    /home/simon/local/opticks_externals/xercesc/lib/pkgconfig
    /home/simon/local/opticks_externals/clhep/lib/pkgconfig
    /home/simon/local/opticks_externals/boost/lib/pkgconfig
    /home/simon/local/opticks/lib64/pkgconfig
    /home/simon/local/opticks/externals/lib/pkgconfig
    /home/simon/local/opticks/externals/lib64/pkgconfig

    === opticks-setup-       skip     append                 PATH /usr/local/cuda-10.1/bin
    === opticks-setup-        add     append                 PATH /home/simon/local/opticks/bin
    === opticks-setup-        add     append                 PATH /home/simon/local/opticks/lib
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /home/simon/local/opticks
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /home/simon/local/opticks/externals
    === opticks-setup-        add     append    CMAKE_PREFIX_PATH /home/blyth/local/opticks/externals/OptiX_650
    === opticks-setup-      nodir     append      PKG_CONFIG_PATH /home/simon/local/opticks/lib/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /home/simon/local/opticks/lib64/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /home/simon/local/opticks/externals/lib/pkgconfig
    === opticks-setup-        add     append      PKG_CONFIG_PATH /home/simon/local/opticks/externals/lib64/pkgconfig
    === opticks-setup-        add     append      LD_LIBRARY_PATH /home/simon/local/opticks/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /home/simon/local/opticks/lib64
    === opticks-setup-        add     append      LD_LIBRARY_PATH /home/simon/local/opticks/externals/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /home/simon/local/opticks/externals/lib64
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /usr/local/cuda-10.1/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /usr/local/cuda-10.1/lib64
    === opticks-setup-      nodir     append      LD_LIBRARY_PATH /home/blyth/local/opticks/externals/OptiX_650/lib
    === opticks-setup-        add     append      LD_LIBRARY_PATH /home/blyth/local/opticks/externals/OptiX_650/lib64

    [simon@localhost cudarap]$ om-conf
    === om-one-or-all conf : cudarap         /home/simon/opticks/cudarap                                  /home/simon/local/opticks/build/cudarap                      
    -- Configuring CUDARap
    -- OpticksCUDAFlags.cmake : COMPUTE_CAPABILITY : 70
    -- OpticksCUDAFlags.cmake : CUDA_NVCC_FLAGS    : -Xcompiler -fPIC;-gencode=arch=compute_70,code=sm_70;-std=c++11;-O2;--use_fast_math 
    -- Use examples/UseOpticksCUDA/CMakeLists.txt for testing FindOpticksCUDA.cmake
    --   CUDA_TOOLKIT_ROOT_DIR   : /usr/local/cuda-10.1 
    --   CUDA_SDK_ROOT_DIR       : CUDA_SDK_ROOT_DIR-NOTFOUND 
    --   CUDA_VERSION            : 10.1 
    --   HELPER_CUDA_INCLUDE_DIR : /usr/local/cuda-10.1/samples/common/inc 
    --   PROJECT_SOURCE_DIR      : /home/simon/opticks/cudarap 
    --   CMAKE_CURRENT_LIST_DIR  : /home/simon/opticks/cmake/Modules 
    -- FindOpticksCUDA.cmake:OpticksCUDA_VERBOSE      : ON 
    -- FindOpticksCUDA.cmake:OpticksCUDA_FOUND        : YES 
    -- FindOpticksCUDA.cmake:OpticksHELPER_CUDA_FOUND : YES 
    -- FindOpticksCUDA.cmake:OpticksCUDA_API_VERSION  : 10010 
    -- FindOpticksCUDA.cmake:CUDA_LIBRARIES           : /usr/local/cuda-10.1/lib64/libcudart_static.a;-lpthread;dl;/usr/lib64/librt.so 
    -- FindOpticksCUDA.cmake:CUDA_INCLUDE_DIRS        : /usr/local/cuda-10.1/include 
    -- FindOpticksCUDA.cmake:CUDA_curand_LIBRARY      : /usr/local/cuda-10.1/lib64/libcurand.so
     key='CUDA_cudart_static_LIBRARY' val='/usr/local/cuda-10.1/lib64/libcudart_static.a' 
     key='CUDA_curand_LIBRARY' val='/usr/local/cuda-10.1/lib64/libcurand.so' 

    -- CUDARap.CUDA_NVCC_FLAGS : -Xcompiler -fPIC;-gencode=arch=compute_70,code=sm_70;-std=c++11;-O2;--use_fast_math 
    -- CUDARap INTERFACE_LINK_LIBRARIES:/usr/local/cuda-10.1/lib64/libcudart_static.a;-lpthread;dl;/usr/lib64/librt.so 
    -- CUDARap.LIBRARIES : Opticks::SysRap;Opticks::OpticksCUDA;Opticks::CUDASamples;Opticks::OKConf;ssl 
    ====== tgt:CUDARap tgt_DIR: ================
    tgt='CUDARap' prop='INTERFACE_INCLUDE_DIRECTORIES' defined='0' set='1' value='$<BUILD_INTERFACE:/home/simon/opticks/cudarap>' 

    tgt='CUDARap' prop='INTERFACE_LINK_LIBRARIES' defined='0' set='1' value='/usr/local/cuda-10.1/lib64/libcudart_static.a;-lpthread;dl;/usr/lib64/librt.so;Opticks::SysRap;Opticks::OpticksCUDA;Opticks::CUDASamples;Opticks::OKConf;ssl' 


    -- bcm_auto_pkgconfig_each LIB:Opticks::CUDASamples : MISSING LIB_PKGCONFIG_NAME 
    -- Configuring CUDARapTest
    -- OpticksCompilationFlags.cmake : CMAKE_BUILD_TYPE = Debug
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS =  -fvisibility=hidden -fvisibility-inlines-hidden -fdiagnostics-show-option -Wall -Wno-unused-function -Wno-comment -Wno-deprecated -Wno-shadow
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_DEBUG = -g
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELEASE = -O2 -DNDEBUG
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_FLAGS_RELWITHDEBINFO= -O2 -g -DNDEBUG
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD : 14 
    -- OpticksCompilationFlags.cmake : CMAKE_CXX_STANDARD_REQUIRED : on 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /home/simon/local/opticks/build/cudarap
    [simon@localhost cudarap]$ 



