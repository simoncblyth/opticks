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

Decide that is too disruptive to do in blyth account so adopt Linux.simon and Darwin.charles accounts for this.


Darwin.charles
-----------------

Common symbolic linked sources between blyth and charles accounts on Darwin:: 

    epsilon:~ charles$ ln -s /Users/blyth/opticks
    epsilon:~ charles$ ln -s /Users/blyth/G4OpticksTest 



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



Darwin  "charles" : x4 link errors not stopping opticks-full
---------------------------------------------------------------

Fixed the om.bash error propagation failure by changing the "while pipe read" into for loop.


Darwin  "charles" : x4 GDML xercesc_3_2 link errors 
---------------------------------------------------------------


::

    ssh A 

    epsilon:~ charles$ opticks-
    epsilon:~ charles$ opticks-full
    ...   
    [ 50%] Linking CXX shared library libExtG4.dylib
    Undefined symbols for architecture x86_64:
      "G4GDMLRead::UserinfoRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLRead::ExtensionRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLWrite::AddExtension(xercesc_3_2::DOMElement*, G4LogicalVolume const*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWrite::UserinfoWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWrite::ExtensionWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLReadSetup::SetupRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadDefine::DefineRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadSolids::SolidsRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLWriteSetup::SetupWrite(xercesc_3_2::DOMElement*, G4LogicalVolume const*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWriteDefine::DefineWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWriteSolids::SolidsWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLReadParamvol::ParamvolRead(xercesc_3_2::DOMElement const*, G4LogicalVolume*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadParamvol::Paramvol_contentRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadMaterials::MaterialsRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadStructure::VolumeRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadStructure::StructureRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLReadStructure::Volume_contentRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLWriteParamvol::ParamvolWrite(xercesc_3_2::DOMElement*, G4VPhysicalVolume const*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWriteParamvol::ParamvolAlgorithmWrite(xercesc_3_2::DOMElement*, G4VPhysicalVolume const*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWriteMaterials::MaterialsWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWriteStructure::StructureWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
    ld: symbol(s) not found for architecture x86_64
    clang: error: linker command failed with exit code 1 (use -v to see invocation)
    make[2]: *** [libExtG4.dylib] Error 1
    make[1]: *** [CMakeFiles/ExtG4.dir/all] Error 2
    make: *** [all] Error 2
    === om-one-or-all install : non-zero rc 2
    === om-all om-install : ERROR bdir /Users/charles/local/opticks/build/extg4 : non-zero rc 2
    === opticks-prepare-installation : generating RNG seeds into installcache
    2020-12-18 20:48:00.036 INFO  [4934099] [main@54]  work 1000000 max_blocks 128 seed 0 offset 0 threads_per_block 256 cachedir /Users/charles/.opticks/rngcache/RNG


::

    -- _lib Geant4::G4persistency _type SHARED_LIBRARY  
    -- _lib G4persistency _loc /usr/local/opticks_externals/g4_1062/lib/libG4persistency.dylib 



Looks like multiple (or updated) xerces-c version inconsistency issue.

::

    epsilon:lib blyth$ otool -L libG4persistency.dylib
    libG4persistency.dylib:
        @rpath/libG4persistency.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4run.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/local/opticks_externals/xercesc/lib/libxerces-c-3.1.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4event.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4tracking.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4processes.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4digits_hits.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4analysis.dylib (compatibility version 0.0.0, current version 0.0.0)
        /opt/local/lib/libexpat.1.dylib (compatibility version 8.0.0, current version 8.11.0)
        @rpath/libG4zlib.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4track.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4particles.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4geometry.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4graphics_reps.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4materials.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4intercoms.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libG4global.dylib (compatibility version 0.0.0, current version 0.0.0)
        @rpath/libCLHEP-2.4.1.0.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.0)
        /usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.50.4)
    epsilon:lib blyth$ 

::

    epsilon:lib blyth$ l /usr/local/opticks_externals/xercesc/lib/
    total 88712
    drwxr-xr-x  4 blyth  staff       128 Jul  3 22:48 pkgconfig
    -rwxr-xr-x  1 blyth  staff      1016 Jun  6  2020 libxerces-c.la
    lrwxr-xr-x  1 blyth  staff        21 Jun  6  2020 libxerces-c.dylib -> libxerces-c-3.1.dylib
    -rwxr-xr-x  1 blyth  staff   4679812 Jun  6  2020 libxerces-c-3.1.dylib
    -rw-r--r--  1 blyth  staff  40730760 Jun  6  2020 libxerces-c.a
    epsilon:lib blyth$ 

    epsilon:lib blyth$ nm /usr/local/opticks_externals/g4_1062/lib/libG4persistency.dylib | c++filt | grep G4GDMLRead::UserinfoRead
    0000000000154410 T G4GDMLRead::UserinfoRead(xercesc_3_1::DOMElement const*)



Looks like the geant4 build used 3_2 possibly from macports but opticks build is using its own 3_1 ?

One of the builds using the macports lib::

    epsilon:lib blyth$ l /opt/local/lib/libxerces-c*
    -rwxr-xr-x  1 root  admin  3262312 Apr 13  2020 /opt/local/lib/libxerces-c-3.2.dylib
    -rw-r--r--  1 root  admin  7056536 Apr 13  2020 /opt/local/lib/libxerces-c.a
    lrwxr-xr-x  1 root  admin       21 Apr 13  2020 /opt/local/lib/libxerces-c.dylib -> libxerces-c-3.2.dylib
    epsilon:lib blyth$ 


::

    epsilon:~ charles$ g4-
    epsilon:~ charles$ g4-cmake-info
    g4-cmake-info
    ===============

       cmake \ 
           -G "Unix Makefiles" \
           -DCMAKE_BUILD_TYPE=Debug \
           -DGEANT4_INSTALL_DATA=ON \ 
           -DGEANT4_USE_GDML=ON \
           -DGEANT4_USE_SYSTEM_CLHEP=ON \ 
           -DGEANT4_INSTALL_DATA_TIMEOUT=3000  \
           -DXERCESC_LIBRARY=/usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib \
           -DXERCESC_INCLUDE_DIR=/usr/local/opticks_externals/xercesc/include \
           -DCMAKE_INSTALL_PREFIX=/usr/local/opticks_externals/g4_1062 \
           /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02                                   


       opticks-cmake-generator : Unix Makefiles
       opticks-buildtype       : Debug
       xercesc-pc-library      : /usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib
       xercesc-pc-includedir   : /usr/local/opticks_externals/xercesc/include
       g4-prefix               : /usr/local/opticks_externals/g4_1062
       g4-dir                  : /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02

    epsilon:~ charles$ 


    epsilon:opticks blyth$ l /usr/local/opticks_externals/xercesc/lib/
    total 88712
    drwxr-xr-x  4 blyth  staff       128 Jul  3 22:48 pkgconfig
    -rwxr-xr-x  1 blyth  staff      1016 Jun  6  2020 libxerces-c.la
    lrwxr-xr-x  1 blyth  staff        21 Jun  6  2020 libxerces-c.dylib -> libxerces-c-3.1.dylib
    -rwxr-xr-x  1 blyth  staff   4679812 Jun  6  2020 libxerces-c-3.1.dylib
    -rw-r--r--  1 blyth  staff  40730760 Jun  6  2020 libxerces-c.a
    epsilon:opticks blyth$ 




Huh: g4--1062 not landing in the expected versioned prefix.::

    epsilon:~ blyth$ g4-
    epsilon:~ blyth$ g4--1062

       g4-dir : /usr/local/opticks_externals/g4.build/geant4.10.06.p02
       g4-nom : geant4.10.06.p02
       g4-url : http://cern.ch/geant4-data/releases/geant4.10.06.p02.tar.gz

    getting http://cern.ch/geant4-data/releases/geant4.10.06.p02.tar.gz
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   180  100   180    0     0   1052      0 --:--:-- --:--:-- --:--:--  1058
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    100 33.2M  100 33.2M    0     0   273k      0  0:02:04  0:02:04 --:--:--  191k
    x geant4.10.06.p02/
    x geant4.10.06.p02/config/
    x geant4.10.06.p02/config/History
    x geant4.10.06.p02/config/analysis.gmk
    x geant4.10.06.p02/config/genwindef.cc
    x geant4.10.06.p02/config/liblist.c
    x geant4.10.06.p02/config/G4UI_USE.gmk
    x geant4.10.06.p02/config/moc.gmk


    ...

    g4-cmake-info
    ===============

       cmake \ 
           -G "Unix Makefiles" \
           -DCMAKE_BUILD_TYPE=Debug \
           -DGEANT4_INSTALL_DATA=ON \ 
           -DGEANT4_USE_GDML=ON \
           -DGEANT4_USE_SYSTEM_CLHEP=ON \ 
           -DGEANT4_INSTALL_DATA_TIMEOUT=3000  \
           -DXERCESC_LIBRARY=/usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib \
           -DXERCESC_INCLUDE_DIR=/usr/local/opticks_externals/xercesc/include \
           -DCMAKE_INSTALL_PREFIX=/usr/local/opticks_externals/g4 \
           /usr/local/opticks_externals/g4.build/geant4.10.06.p02                                   



       opticks-cmake-generator : Unix Makefiles
       opticks-buildtype       : Debug
       xercesc-pc-library      : /usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib
       xercesc-pc-includedir   : /usr/local/opticks_externals/xercesc/include
       g4-prefix               : /usr/local/opticks_externals/g4
       g4-dir                  : /usr/local/opticks_externals/g4.build/geant4.10.06.p02

    -- The C compiler identification is AppleClang 9.0.0.9000039
    -- The CXX compiler identification is AppleClang 9.0.0.9000039
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc
    -- Check for working C compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/cc - works
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Check for working CXX compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++
    -- Check for working CXX compiler: /Applications/Xcode/Xcode_9_2.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ - works
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Found EXPAT: /opt/local/lib/libexpat.dylib (found version "2.0.1") 
    -- Found XercesC: /usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib (found version "3.1.1") 
    -- Configuring download of missing dataset G4NDL (4.6)
    -- Configuring download of missing dataset G4EMLOW (7.9.1)
    -- Configuring download of missing dataset PhotonEvaporation (5.5)
    ...

    -- Installing: /usr/local/opticks_externals/g4/include/Geant4/G4VModelCommand.hh
    -- Installing: /usr/local/opticks_externals/g4/include/Geant4/G4VModelFactory.hh
    -- Installing: /usr/local/opticks_externals/g4/include/Geant4/G4VTrajectoryModel.hh
    -- Installing: /usr/local/opticks_externals/g4/include/Geant4/G4VisTrajContext.hh
    -- Installing: /usr/local/opticks_externals/g4/include/Geant4/G4VisTrajContext.icc
    Sat Dec 19 17:18:31 GMT 2020
    generate /usr/local/opticks_externals/g4/lib/pkgconfig/Geant4.pc
    epsilon:geant4.10.06.p02.Debug.build blyth$ g4-
    epsilon:geant4.10.06.p02.Debug.build blyth$ g4-prefix
    /usr/local/opticks_externals/g4
    epsilon:geant4.10.06.p02.Debug.build blyth$ 
                     


    epsilon:geant4.10.06.p02.Debug.build blyth$ echo $OPTICKS_GEANT4_PREFIX
    /usr/local/opticks_externals/g4
    epsilon:geant4.10.06.p02.Debug.build blyth$ t g4-prefix
    g4-prefix () 
    { 
        echo ${OPTICKS_GEANT4_PREFIX:-$(opticks-prefix)_externals/g4_$(g4-ver)}
    }
    epsilon:geant4.10.06.p02.Debug.build blyth$ 


Tripped up by having run opticks-setup.sh which sets the envvar::

    epsilon:issues blyth$ mdfind OPTICKS_GEANT4_PREFIX
    /Users/blyth/opticks/externals/g4.bash
    /Users/charles/local/opticks/bin/opticks-setup.sh
    /Users/blyth/opticks/opticks.bash
    /Users/blyth/junotop/offline/installation/junoenv/packages/opticks.sh
    /Users/blyth/junotop/junoenv/packages/opticks.sh
    /Users/blyth/junotop/ExternalLibs/Opticks/0.0.0-rc1/bin/opticks-setup.sh
    /Users/blyth/junotop/ExternalLibs/Build/opticks-0.0.0-rc1/opticks.bash
    /Users/blyth/junotop/ExternalLibs/Build/opticks-0.0.0-rc1/externals/g4.bash
    epsilon:issues blyth$ 


    271 # opticks-setup-geant4-  
    272 
    273 export OPTICKS_GEANT4_PREFIX=$(opticks-setup-find-geant4-prefix)
    274 
    275 if [ -n "$OPTICKS_GEANT4_PREFIX" ]; then
    276     if [ -f "$OPTICKS_GEANT4_PREFIX/bin/geant4.sh" ]; then
    277         source $OPTICKS_GEANT4_PREFIX/bin/geant4.sh
    278     else
    279         echo ERROR no $OPTICKS_GEANT4_PREFIX/bin/geant4.sh at OPTICKS_GEANT4_PREFIX : $OPTICKS_GEANT4_PREFIX
    280         return 1
    281     fi
    282 fi


Having Geant4 already in CMAKE_PREFIX_PATH trips up the destination of 
a subsequent g4-prefix::

    epsilon:issues blyth$ t opticks-setup-find-geant4-prefix
    opticks-setup-find-geant4-prefix () 
    { 
        opticks-setup-find-config-prefix Geant4
    }
    epsilon:issues blyth$ t opticks-setup-find-config-prefix
    opticks-setup-find-config-prefix () 
    { 
        : mimick CMake "find_package name CONFIG" identifing the first prefix in the path;
        local name=${1:-Geant4};
        local prefix="";
        local ifs=$IFS;
        IFS=:;
        for pfx in $CMAKE_PREFIX_PATH;
        do
            ls -1 $pfx/lib*/$name-*/${name}Config.cmake 2> /dev/null 1>&2;
            [ $? -eq 0 ] && prefix=$pfx && break;
            ls -1 $pfx/lib*/cmake/$name-*/${name}Config.cmake 2> /dev/null 1>&2;
            [ $? -eq 0 ] && prefix=$pfx && break;
        done;
        IFS=$ifs;
        echo $prefix
    }
    epsilon:issues blyth$ 


The mistake is the g4 line 25 below::

     12 # PATH envvars control the externals that opticks/CMake or pkg-config will find  
     13 unset CMAKE_PREFIX_PATH
     14 unset PKG_CONFIG_PATH
     15 
     16 # mandatory envvars in buildenv 
     17 export OPTICKS_PREFIX=/usr/local/opticks
     18 export OPTICKS_CUDA_PREFIX=/usr/local/cuda
     19 export OPTICKS_OPTIX_PREFIX=/usr/local/optix
     20 export OPTICKS_COMPUTE_CAPABILITY=30
     21        
     22 ## hookup paths to access "foreign" externals 
     23 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     24 opticks-prepend-prefix /usr/local/opticks_externals/xercesc
     25 opticks-prepend-prefix /usr/local/opticks_externals/g4
     26 opticks-prepend-prefix /usr/local/opticks_externals/boost
     27 
     28 # non-standard 
     29 #opticks-prepend-prefix /usr/local/opticks_externals/zeromq-4.0.4
     30 
     31 source $OPTICKS_PREFIX/bin/opticks-setup.sh 1> /dev/null
     32 



Add *g4-check-no-prior-prefix* to avoid stomping on a prior prefix::

     606 g4--()
     607 {   
     608     local msg="=== $FUNCNAME :"
     609     g4-check-no-prior-prefix
     610     [ $? -ne 0 ] && echo $msg check-prior-prefix FAIL && return 1
     611     g4-get
     612     [ $? -ne 0 ] && echo $msg get FAIL && return 1
     613     g4-configure 
     614     [ $? -ne 0 ] && echo $msg configure FAIL && return 2
     615     g4-build 
     616     [ $? -ne 0 ] && echo $msg build FAIL && return 3
     617     # g4-export-ini
     618     # [ $? -ne 0 ] && echo $msg export-ini FAIL && return 4
     619     g4-pc
     620     [ $? -ne 0 ] && echo $msg pc FAIL && return 5
     621     
     622     return 0
     623 }
     624 
     625 
     626 g4-check-no-prior-prefix()
     627 {
     628     local msg="=== $FUNCNAME :"
     629     local prior=$(opticks-setup-find-geant4-prefix)
     630     local rc 
     631     if [ "$prior" == "" ]; then
     632         rc=0
     633     else
     634         echo $msg prior prefix found : $prior : remove geant4 prefix from CMAKE_PREFIX_PATH and or remove the prefix dir and try again 
     635         rc=1
     636     fi
     637     return $rc
     638 }


::

    epsilon:issues blyth$ g4-
    epsilon:issues blyth$ g4--
    === g4-check-no-prior-prefix : prior prefix found : /usr/local/opticks_externals/g4 : remove geant4 prefix from CMAKE_PREFIX_PATH and or remove the prefix dir and try again
    === g4-- : check-prior-prefix FAIL
    epsilon:issues blyth$ 


    epsilon:issues blyth$ opticks-setup-path
    /usr/local/opticks/bin/opticks-setup.sh
    epsilon:issues blyth$ vi /usr/local/opticks/bin/opticks-setup.sh


    259 # opticks-setup-libpaths-  
    260 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_PREFIX/lib
    261 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_PREFIX/lib64
    262 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_PREFIX/externals/lib
    263 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_PREFIX/externals/lib64
    264 
    265 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_CUDA_PREFIX/lib
    266 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_CUDA_PREFIX/lib64
    267 
    268 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_OPTIX_PREFIX/lib
    269 opticks-setup- append DYLD_LIBRARY_PATH $OPTICKS_OPTIX_PREFIX/lib64
    270 
    271 # opticks-setup-geant4-  
    272 
    273 export OPTICKS_GEANT4_PREFIX=$(opticks-setup-find-geant4-prefix)
    274 
    275 if [ -n "$OPTICKS_GEANT4_PREFIX" ]; then
    276     if [ -f "$OPTICKS_GEANT4_PREFIX/bin/geant4.sh" ]; then
    277         source $OPTICKS_GEANT4_PREFIX/bin/geant4.sh
    278     else
    279         echo ERROR no $OPTICKS_GEANT4_PREFIX/bin/geant4.sh at OPTICKS_GEANT4_PREFIX : $OPTICKS_GEANT4_PREFIX
    280         return 1
    281     fi
    282 fi



Introduce some safety measures in g4- to prevent stomping on prior g4-prefix.
Move to always using a versioned prefix.


1062 is failing to download some data, with multiple timeouts::

    [  2%] Performing download step (download, verify and extract) for 'G4NDL'
    -- verifying file...
           file='/usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz'
    -- MD5 hash of
        /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz
      does not match expected value
        expected: 'd07e43499f607e01f2c1ce06d7a09f3e'
          actual: '56f7e0a2835afe18d156f2722b99615e'
    -- File already exists but hash mismatch. Removing...
    -- Downloading...
       dst='/usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz'
       timeout='100000 seconds'
    -- Using src='https://cern.ch/geant4-data/datasets/G4NDL.4.6.tar.gz'
    -- [download 100% complete]
    -- [download 0% complete]
    -- [download 1% complete]
    -- [download 2% complete]
    -- [download 3% complete]
    -- [download 4% complete]
    -- Retrying...
    -- Using src='https://cern.ch/geant4-data/datasets/G4NDL.4.6.tar.gz'
    -- [download 100% complete]
    -- [download 0% complete]
    -- [download 1% complete]
    -- [download 2% complete]
    -- [download 3% complete]
    -- [download 4% complete]
    -- [download 5% complete]

    epsilon:issues blyth$ l /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/
    total 295040
    -rw-r--r--  1 blyth  staff  138018816 Dec 19 23:15 G4NDL.4.6.tar.gz
    drwxr-xr-x  7 blyth  staff        224 Dec 19 22:49 G4NDL-stamp
    drwxr-xr-x  2 blyth  staff         64 Dec 19 21:32 G4NDL-build
    epsilon:issues blyth$ rm /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz
    epsilon:issues blyth$ 


Grab it from IHEP using scp::

   scp P:local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02.Debug.build/Externals/G4NDL-4.6/src/G4NDL.4.6.tar.gz 

   ETA > 1hr 




Darwin.charles : opticks-full-make x4 runs into same xercesc_3_2 xercesc_3_1 issue even after rebuilding g4_1062
------------------------------------------------------------------------------------------------------------------

::
    

    opticks-full-make
    ...


    -- Adding boost_regex dependencies: headers
    -- FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments 
    -- CLHEP_DIR : /usr/local/opticks_externals/clhep/lib/CLHEP-2.4.1.0
    -- CLHEP_INCLUDE_DIRS : /usr/local/opticks_externals/clhep/lib/CLHEP-2.4.1.0/../../include
    -- CLHEP_LIBRARIES    : CLHEP::CLHEP
    -- bcm_auto_pkgconfig_each LIB:CLHEP::CLHEP : MISSING LIB_PKGCONFIG_NAME 
    -- Configuring ExtG4Test
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /Users/charles/local/opticks/build/extg4
    === om-make-one : extg4           /Users/charles/opticks/extg4                                 /Users/charles/local/opticks/build/extg4                     
    [  1%] Linking CXX shared library libExtG4.dylib
    Undefined symbols for architecture x86_64:
      "G4GDMLRead::UserinfoRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLRead::ExtensionRead(xercesc_3_2::DOMElement const*)", referenced from:
          vtable for X4GDMLReadStructure in X4GDMLReadStructure.cc.o
      "G4GDMLWrite::AddExtension(xercesc_3_2::DOMElement*, G4LogicalVolume const*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o
      "G4GDMLWrite::UserinfoWrite(xercesc_3_2::DOMElement*)", referenced from:
          vtable for X4GDMLWriteStructure in X4GDMLWriteStructure.cc.o


Notice::

    -- FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments 

Looks like the Opticks build is trying to do something "clever" with G4persistency target : that presumably no longer works in 1062.


Darwin.blyth add some debug to FindOpticksXercesC.cmake 
----------------------------------------------------------

x4/CMakeLists.txt::

     32 # just for X4GDMLWrite
     33 set(OpticksXercesC_VERBOSE ON)
     34 find_package(OpticksXercesC REQUIRED MODULE)

::

    x4
    om-conf
    ...
    -- Adding boost_regex dependencies: headers
    -- OpticksXercesC_MODULE : /Users/blyth/opticks/cmake/Modules/FindOpticksXercesC.cmake 
    -- FindOpticksXercesC.cmake. Found G4persistency target _lll G4geometry;G4global;G4graphics_reps;G4intercoms;G4materials;G4particles;G4digits_hits;G4event;G4processes;G4run;G4track;G4tracking;/usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib
    --  G4persistency.xercesc_lib         : /usr/local/opticks_externals/xercesc/lib/libxerces-c.dylib 
    --  G4persistency.xercesc_include_dir : /usr/local/opticks_externals/xercesc/include 
     


Darwin.charles  shares same opticks source (via symbolic link) as blyth by different CMAKE_PREFIX_PATH
----------------------------------------------------------------------------------------------------------


::

    -- OpticksXercesC_MODULE : /Users/charles/opticks/cmake/Modules/FindOpticksXercesC.cmake 
    -- FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments 
    -- FindOpticksXercesC.cmake OpticksXercesC_MODULE      : /Users/charles/opticks/cmake/Modules/FindOpticksXercesC.cmake  
    -- FindOpticksXercesC.cmake OpticksXercesC_INCLUDE_DIR : /opt/local/include  
    -- FindOpticksXercesC.cmake OpticksXercesC_LIBRARY     : /opt/local/lib/libxerces-c.dylib  
    -- FindOpticksXercesC.cmake OpticksXercesC_FOUND       : YES  



Make the G4persistency target fishing work with Geant4 1062
-------------------------------------------------------------




::

     42 set(xercesc_lib)
     43 set(xercesc_include_dir)
     44    
     45 if(TARGET Geant4::G4persistency AND TARGET XercesC::XercesC)
     46    # this works with Geant4 1062
     47    get_target_property(_lll Geant4::G4persistency INTERFACE_LINK_LIBRARIES)
     48    message(STATUS "FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll ${_lll} " )
     49    
     50    get_target_property(xercesc_lib         XercesC::XercesC IMPORTED_LOCATION )
     51    get_target_property(xercesc_include_dir XercesC::XercesC INTERFACE_INCLUDE_DIRECTORIES )
     52    
     53    if(OpticksXercesC_VERBOSE)
     54        message(STATUS "FindOpticksXercesC.cmake. XercesC::XercesC target xercesc_lib         : ${xercesc_lib} " )
     55        message(STATUS "FindOpticksXercesC.cmake. XercesC::XercesC target xercesc_include_dir : ${xercesc_include_dir} " )
     56    endif()
     57    
     58    
     59 elseif(TARGET G4persistency)
     60    # this works with Geant4 1042
     61     get_target_property(_lll G4persistency INTERFACE_LINK_LIBRARIES)
     62     message(STATUS "FindOpticksXercesC.cmake. Found G4persistency target _lll ${_lll}" )
     63     foreach(_lib ${_lll})
     64         get_filename_component(_nam ${_lib} NAME)
     65         string(FIND "${_nam}" "libxerces-c" _pos )
     66         if(_pos EQUAL 0)
     67             #message(STATUS "_lib ${_lib}  _nam ${_nam} _pos ${_pos} ") 
     68             set(xercesc_lib ${_lib})
     69         endif()
     70     endforeach()
     71     
     72     if(xercesc_lib)
     73         get_filename_component(_dir ${xercesc_lib} DIRECTORY)
     74         get_filename_component(_dirdir ${_dir} DIRECTORY) 
     75         set(xercesc_include_dir "${_dirdir}/include" )    
     76     endif()
     77     
     78     if(OpticksXercesC_VERBOSE)
     79        message(STATUS " G4persistency.xercesc_lib         : ${xercesc_lib} ")
     80        message(STATUS " G4persistency.xercesc_include_dir : ${xercesc_include_dir} ")
     81     endif()
     82     
     83 else()
     84     #message(FATAL_ERROR "G4persistency target is required" )
     85     message(STATUS "FindOpticksXercesC.cmake. Did not find G4persistency target : so look for system XercesC or one provided by cmake arguments " )
     86 endif()
     87 






Darwin.charles.1062 CUDA CMake warning
-----------------------------------------

::

    -- FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll Geant4::G4geometry;Geant4::G4global;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4materials;Geant4::G4particles;Geant4::G4digits_hits;Geant4::G4event;Geant4::G4processes;Geant4::G4run;Geant4::G4track;Geant4::G4tracking;XercesC::XercesC 
    -- FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll Geant4::G4geometry;Geant4::G4global;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4materials;Geant4::G4particles;Geant4::G4digits_hits;Geant4::G4event;Geant4::G4processes;Geant4::G4run;Geant4::G4track;Geant4::G4tracking;XercesC::XercesC 
    CMake Warning (dev) at /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake:590 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
    Call Stack (most recent call first):
      /Users/charles/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/cudarap/cudarap-config.cmake:16 (find_dependency)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/thrustrap/thrustrap-config.cmake:16 (find_dependency)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    CMake Warning (dev) at /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake:596 (option):
      Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
      --help-policy CMP0077" for policy details.  Use the cmake_policy command to
      set the policy and suppress this warning.

      For compatibility with older versions of CMake, option is clearing the
      normal variable 'CUDA_VERBOSE_BUILD'.
    Call Stack (most recent call first):
      /Users/charles/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/cudarap/cudarap-config.cmake:16 (find_dependency)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/thrustrap/thrustrap-config.cmake:16 (find_dependency)
      /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
      /Users/charles/local/opticks/lib/cmake/cfg4/cfg4-config.cmake:16 (find_dependency)
      CMakeLists.txt:11 (find_package)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Found CUDA: /usr/local/cuda (found version "9.1") 
    -- FindOpticksXercesC.cmake. Found Geant4::G4persistency AND XercesC::XercesC target _lll Geant4::G4geometry;Geant4::G4global;Geant4::G4graphics_reps;Geant4::G4intercoms;Geant4::G4materials;Geant4::G4particles;Geant4::G4digits_hits;Geant4::G4event;Geant4::G4processes;Geant4::G4run;Geant4::G4track;Geant4::G4tracking;XercesC::XercesC 
    -- Configuring G4OKTest
    -


Darwin.charles.1062 geocache-create assert
--------------------------------------------

::

    2020-12-20 15:03:49.913 INFO  [6523967] [Opticks::loadOriginCacheMeta@1886]  cachemetapath /Users/charles/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/cachemeta.json
    2020-12-20 15:03:49.913 INFO  [6523967] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    2020-12-20 15:03:49.913 FATAL [6523967] [Opticks::ExtractCacheMetaGDMLPath@2053]  FAILED TO EXTRACT ORIGIN GDMLPATH FROM METADATA argline 
     argline -
    2020-12-20 15:03:49.913 INFO  [6523967] [Opticks::loadOriginCacheMeta@1897] ExtractCacheMetaGDMLPath 
    2020-12-20 15:03:49.913 FATAL [6523967] [Opticks::loadOriginCacheMeta@1903] cachemetapath /Users/charles/.opticks/geocache/OKX4Test_World0xc15cfc00x40f7000_PV_g4live/g4ok_gltf/50a18baaf29b18fae8c1642927003ee3/1/cachemeta.json
    2020-12-20 15:03:49.913 FATAL [6523967] [Opticks::loadOriginCacheMeta@1904] argline that creates cachemetapath must include "--gdmlpath /path/to/geometry.gdml" 
    Assertion failed: (m_origin_gdmlpath), function loadOriginCacheMeta, file /Users/charles/opticks/optickscore/Opticks.cc, line 1906.
    /Users/charles/local/opticks/bin/o.sh: line 362: 41266 Abort trap: 6           /Users/charles/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/charles/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    === o-main : runline PWD /tmp/charles/opticks/geocache-create- RC 134 Sun Dec 20 15:03:49 GMT 2020
    /Users/charles/local/opticks/lib/OKX4Test --okx4test --g4codegen --deletegeocache --gdmlpath /Users/charles/local/opticks/opticksaux/export/DayaBay_VGDX_20140414-1300/g4_00_CGeometry_export_v1.gdml --x4polyskip 211,232 --geocenter --noviz --runfolder geocache-dx1 --runcomment sensors-gdml-review.rst
    echo o-postline : dummy
    o-postline : dummy
    /Users/charles/local/opticks/bin/o.sh : RC : 134
    epsilon:cfg4 charles$ 


geocache-create -D::

    (lldb) bt
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010d101d8e libOpticksCore.dylib`Opticks::loadOriginCacheMeta(this=0x00000001198e6e00) at Opticks.cc:1906
        frame #5: 0x000000010d1075ef libOpticksCore.dylib`Opticks::postconfigure(this=0x00000001198e6e00) at Opticks.cc:2445
        frame #6: 0x000000010d106e68 libOpticksCore.dylib`Opticks::configure(this=0x00000001198e6e00) at Opticks.cc:2404
        frame #7: 0x000000010001564c OKX4Test`main(argc=15, argv=0x00007ffeefbfed58) at OKX4Test.cc:95


::

    epsilon:cfg4 charles$ Opticks=INFO geocache-create -D

    ...
    2020-12-20 15:10:15.295 INFO  [6534489] [BMeta::dump@199] Opticks::loadOriginCacheMeta
    2020-12-20 15:10:15.295 INFO  [6534489] [Opticks::ExtractCacheMetaGDMLPath@2013]  argline -
    2020-12-20 15:10:15.295 INFO  [6534489] [Opticks::ExtractCacheMetaGDMLPath@2041] 


Ahha : Opticks is assuming that the geocache and metadata exists already but that is not true when running 
geocache-create for first time. In that situation should be parsing the current executable commandline not the persisted argline.

Could assume that lack of an OPTICKS_KEY envvar signals creation of geocache ?


::

    1335 /**
    1336 Opticks::isKeySource
    1337 ----------------------
    1338 
    1339 Name of current executable matches that of the creator of the geocache.
    1340 BUT what about the first run of geocache-create ?
    1341 
    1342 **/
    1343 
    1344 bool Opticks::isKeySource() const
    1345 {
    1346     return m_rsc->isKeySource();
    1347 }   
    1348 

    1240 bool BOpticksResource::isKeySource() const   // name of current executable matches that of the creator of the geocache
    1241 {
    1242     return m_key ? m_key->isKeySource() : false ;
    1243 }

    094 bool BOpticksKey::isKeySource() const  // current executable is geocache creator 
     95 {
     96     return m_current_exename && m_exename && strcmp(m_current_exename, m_exename) == 0 ;
     97 }




How to distinguish geocache and key creation as done by geocache-create from consumption ?  
The difference is that a spec is obtained from translated geometry.

okg4:tests/OKX4Test.cc::

     74     BMeta* auxmeta = NULL ;
     75     G4VPhysicalVolume* top = CGDML::Parse( gdmlpath, &auxmeta ) ;
     76     if( top == NULL ) return 0 ;
     77     if(auxmeta) auxmeta->dump("auxmeta");
     78 
     79 
     80     if(PLOG::instance->has_arg("--earlyexit"))
     81     {
     82         LOG(info) << " --earlyexit " ;
     83         return 0 ;
     84     }
     85 
     86 
     87     const char* digestextra1 = csgskiplv ;    // kludge the digest to be sensitive to csgskiplv
     88     const char* spec = X4PhysicalVolume::Key(top, digestextra1, digestextra2 ) ;
     89 
     90     Opticks::SetKey(spec);
     91 
     92     const char* argforce = "--tracer --nogeocache --xanalytic" ;   // --nogeoache to prevent GGeo booting from cache 
     93 
     94     Opticks* m_ok = new Opticks(argc, argv, argforce);  // Opticks instanciation must be after Opticks::SetKey
     95     m_ok->configure();
     96     m_ok->enforceNoGeoCache();


Opticks needs to be aware of a live spec versus a canned one.


g4 1062 with DYB geom has issue with surface conversion
-----------------------------------------------------------

::

    2020-12-20 19:13:31.787 INFO  [6787389] [GMaterialLib::dumpSensitiveMaterials@1230] X4PhysicalVolume::convertMaterials num_sensitive_materials 1
     0 :                       Bialkali
    Assertion failed: (_REFLECTIVITY && os && "non-sensor surfaces must have a reflectivity "), function createStandardSurface, file /Users/charles/opticks/ggeo/GSurfaceLib.cc, line 595.
    Process 73978 stopped
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
        frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
    libsystem_kernel.dylib`__pthread_kill:
    ->  0x7fff77d74b66 <+10>: jae    0x7fff77d74b70            ; <+20>
        0x7fff77d74b68 <+12>: movq   %rax, %rdi
        0x7fff77d74b6b <+15>: jmp    0x7fff77d6bae9            ; cerror_nocancel
        0x7fff77d74b70 <+20>: retq   
    Target 0: (OKX4Test) stopped.

    Process 73978 launched: '/Users/charles/local/opticks/lib/OKX4Test' (x86_64)
    (lldb) bt
    * thread #1, queue = 'com.apple.main-thread', stop reason = signal SIGABRT
      * frame #0: 0x00007fff77d74b66 libsystem_kernel.dylib`__pthread_kill + 10
        frame #1: 0x00007fff77f3f080 libsystem_pthread.dylib`pthread_kill + 333
        frame #2: 0x00007fff77cd01ae libsystem_c.dylib`abort + 127
        frame #3: 0x00007fff77c981ac libsystem_c.dylib`__assert_rtn + 320
        frame #4: 0x000000010cc0be84 libGGeo.dylib`GSurfaceLib::createStandardSurface(this=0x0000000111b17690, src=0x0000000111a68690) at GSurfaceLib.cc:595
        frame #5: 0x000000010cc0ae42 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, surf=0x0000000111a68690) at GSurfaceLib.cc:486
        frame #6: 0x000000010cc0ad84 libGGeo.dylib`GSurfaceLib::addBorderSurface(this=0x0000000111b17690, surf=0x0000000111a68690, pv1="/dd/Geometry/PMT/lvPmtHemi#pvPmtHemiVacuum0xc1340e80x3ee9ae0", pv2="/dd/Geometry/PMT/lvPmtHemiVacuum#pvPmtHemiCathode0xc02c3800x3ee9720", direct=false) at GSurfaceLib.cc:374
        frame #7: 0x000000010cc0aa48 libGGeo.dylib`GSurfaceLib::add(this=0x0000000111b17690, raw=0x0000000111a68690) at GSurfaceLib.cc:358
        frame #8: 0x00000001038ba51e libExtG4.dylib`X4LogicalBorderSurfaceTable::init(this=0x00007ffeefbfd478) at X4LogicalBorderSurfaceTable.cc:66
        frame #9: 0x00000001038ba1d4 libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:45
        frame #10: 0x00000001038ba18d libExtG4.dylib`X4LogicalBorderSurfaceTable::X4LogicalBorderSurfaceTable(this=0x00007ffeefbfd478, dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:44
        frame #11: 0x00000001038ba15c libExtG4.dylib`X4LogicalBorderSurfaceTable::Convert(dst=0x0000000111b17690) at X4LogicalBorderSurfaceTable.cc:37
        frame #12: 0x00000001038c6f63 libExtG4.dylib`X4PhysicalVolume::convertSurfaces(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:282
        frame #13: 0x00000001038c670f libExtG4.dylib`X4PhysicalVolume::init(this=0x00007ffeefbfe558) at X4PhysicalVolume.cc:192
        frame #14: 0x00000001038c63f5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:177
        frame #15: 0x00000001038c56b5 libExtG4.dylib`X4PhysicalVolume::X4PhysicalVolume(this=0x00007ffeefbfe558, ggeo=0x0000000111b14760, top=0x0000000118d44660) at X4PhysicalVolume.cc:168
        frame #16: 0x0000000100015707 OKX4Test`main(argc=15, argv=0x00007ffeefbfed58) at OKX4Test.cc:108
        frame #17: 0x00007fff77c24015 libdyld.dylib`start + 1
        frame #18: 0x00007fff77c24015 libdyld.dylib`start + 1


* :doc:`g4-1062-geocache-create-reflectivity-assert.rst`


g4_1062 changes
----------------

::

    epsilon:~ blyth$ g4-cd
    epsilon:geant4.10.06.p02 blyth$ 

    epsilon:geant4.10.06.p02 blyth$ pwd
    /usr/local/opticks_externals/g4_1062.build/geant4.10.06.p02

    epsilon:geant4.10.06.p02 blyth$ find . -name '*.orig'
    ./source/processes/electromagnetic/xrays/include/G4Cerenkov.hh.orig
    ./source/persistency/gdml/src/G4GDMLReadSolids.cc.orig

    epsilon:geant4.10.06.p02 blyth$ diff source/processes/electromagnetic/xrays/include/G4Cerenkov.hh.orig source/processes/electromagnetic/xrays/include/G4Cerenkov.hh
    199a200
    > public:

    epsilon:geant4.10.06.p02 blyth$ diff source/persistency/gdml/src/G4GDMLReadSolids.cc.orig source/persistency/gdml/src/G4GDMLReadSolids.cc
    2548c2548
    < 	 mapOfMatPropVects[Strip(name)] = propvect;
    ---
    > 	 //mapOfMatPropVects[Strip(name)] = propvect;  //SCB:opticks/extg4/tests/G4GDMLReadSolids_1062_mapOfMatPropVects_bug.cc



    epsilon:geant4.10.06.p02 blyth$ g4-build
    Thu Dec 24 10:20:12 GMT 2020
    [  0%] Built target G4ENSDFSTATE
    [  0%] Built target G4INCL
    ...
    [ 18%] Built target G4geometry
    [ 24%] Built target G4particles
    [ 24%] Built target G4track
    [ 27%] Built target G4digits_hits
    Scanning dependencies of target G4processes
    [ 27%] Building CXX object source/processes/CMakeFiles/G4processes.dir/electromagnetic/xrays/src/G4Cerenkov.cc.o
    [ 27%] Linking CXX shared library ../../BuildProducts/lib/libG4processes.dylib
    [ 82%] Built target G4processes
    [ 83%] Built target G4tracking

