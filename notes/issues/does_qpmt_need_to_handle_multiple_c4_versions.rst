does_qpmt_need_to_handle_multiple_c4_versions
===============================================

Thoughts
-----------

If c4 were only used with Opticks then I would not 
bother but c4 Stack calc is used both in CPU and GPU. 
This suggests it might be best to have c4 version branching. 

But the API change only has impact within C4CustomART.h 
from point of view of junosw just beed to bump the 
version and rebuild. 


Issue on workstation
----------------------

::

    [  1%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QPMT.cu.o
    /data/blyth/junotop/opticks/qudarap/qpmt.h(184): error: no instance of constructor "Stack<T, N>::Stack [with T=float, N=4]" matches the argument list
                argument types are: (float, float, float, float [16], unsigned int)
              detected during:
                instantiation of "void qpmt<F>::get_lpmtid_LL(F *, int, F, F, F) const [with F=float]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(244): here
                instantiation of "void _QPMT_mct_lpmtid<F,P>(qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float, P=16]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(282): here
                instantiation of "void QPMT_mct_lpmtid(dim3, dim3, qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(302): here

    /data/blyth/junotop/opticks/qudarap/qpmt.h(185): error: class "Layr<float>" has no member "cdata"
              detected during:
                instantiation of "void qpmt<F>::get_lpmtid_LL(F *, int, F, F, F) const [with F=float]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(244): here
                instantiation of "void _QPMT_mct_lpmtid<F,P>(qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float, P=16]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(282): here
                instantiation of "void QPMT_mct_lpmtid(dim3, dim3, qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float]" 
    /data/blyth/junotop/opticks/qudarap/QPMT.cu(302): here






TODO : make c4/build.sh install into same dirs as $JUNOTOP/junoenv/packages/custom4.sh
----------------------------------------------------------------------------------------

This should make it possible to reproduce the issue on laptop by making c4 build installdir 


c4/build.sh::

    -- Installing: /usr/local/opticks/lib/libCustom4.dylib
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4OpBoundaryProcess.hh
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4IPMTAccessor.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4CustomART.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4CustomART_Debug.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4MultiLayrStack.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4Touchable.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4TrackInfo.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4Track.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4Pho.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4GS.h
    -- Up-to-date: /usr/local/opticks/include/Custom4/C4Sys.h
    -- Installing: /usr/local/opticks/lib/Custom4-0.1.5/Custom4Config.cmake
    -- Installing: /usr/local/opticks/lib/Custom4-0.1.5/Custom4ConfigVersion.cmake
    -- Installing: /usr/local/opticks/lib/Custom4-0.1.5/C4Version.h



what qu find_package custom4 reports::

    -- Custom4_VERBOSE       : ON 
    -- Custom4_FOUND         : YES 
    -- Custom4_VERSION       : 0.1.4 
    -- Custom4_PREFIX        : /data/blyth/junotop/ExternalLibs/custom4/0.1.4 
    -- Custom4_INCLUDE_DIR   : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/include/Custom4 
    -- Custom4_INCLUDE_DIRS  : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/include/Custom4 
    -- Custom4_CFLAGS        : -I/data/blyth/junotop/ExternalLibs/custom4/0.1.4/include/Custom4 
    -- Custom4_DEFINITIONS   : -DWITH_CUSTOM4 
    -- Custom4_LIBRARY_DIR   : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64 
    -- Custom4_LIBRARY_PATH  : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64/libCustom4.so 
    -- Custom4_LIBRARIES     : -L/data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64 -lCustom4 
    -- Custom4_CMAKE_PATH    : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64/Custom4-0.1.4/Custom4Config.cmake 
    -- Custom4_CMAKE_DIR     : /data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64/Custom4-0.1.4 

    N[blyth@localhost qudarap]$ l /data/blyth/junotop/ExternalLibs/custom4/0.1.4/include/Custom4/
    total 92
     0 drwxrwxr-x. 2 blyth blyth   237 Apr 11 02:54 .
     0 drwxrwxr-x. 3 blyth blyth    21 Apr 11 02:54 ..
     4 -rw-r--r--. 1 blyth blyth   718 Apr 11 02:48 C4CustomART_Debug.h
    12 -rw-r--r--. 1 blyth blyth 11533 Apr 11 02:48 C4CustomART.h
     4 -rw-r--r--. 1 blyth blyth  2079 Apr 11 02:48 C4GS.h
     4 -rw-r--r--. 1 blyth blyth   453 Apr 11 02:48 C4IPMTAccessor.h
    24 -rw-r--r--. 1 blyth blyth 23870 Apr 11 02:48 C4MultiLayrStack.h
    12 -rw-r--r--. 1 blyth blyth 12262 Apr 11 02:48 C4OpBoundaryProcess.hh
     8 -rw-r--r--. 1 blyth blyth  6105 Apr 11 02:48 C4Pho.h
     4 -rw-r--r--. 1 blyth blyth   442 Apr 11 02:48 C4Sys.h
     8 -rw-r--r--. 1 blyth blyth  5415 Apr 11 02:48 C4Touchable.h
     8 -rw-r--r--. 1 blyth blyth  5483 Apr 11 02:48 C4Track.h
     4 -rw-r--r--. 1 blyth blyth  3831 Apr 11 02:48 C4TrackInfo.h

    N[blyth@localhost qudarap]$ l /data/blyth/junotop/ExternalLibs/custom4/0.1.4/lib64/
    total 92
     0 drwxrwxr-x. 3 blyth blyth    48 May 24 19:35 .
    92 -rwxr-xr-x. 1 blyth blyth 92296 May 24 19:35 libCustom4.so
     0 drwxrwxr-x. 4 blyth blyth    62 Apr 11 02:54 ..
     0 drwxrwxr-x. 2 blyth blyth    86 Apr 11 02:54 Custom4-0.1.4


$JUNOTOP/junoenv/packages/custom4.sh::

    143 function juno-ext-libs-custom4-conf- {
    144     # begin to configure
    145     if [ ! -d "custom4-build" ]; then
    146         mkdir custom4-build
    147     fi
    148     pushd custom4-build
    149     cmake .. -DCMAKE_INSTALL_PREFIX:PATH=$(juno-ext-libs-custom4-install-dir)
    150 

So the CMAKE_INSTALL_PREFIX is::

    /data/blyth/junotop/ExternalLibs/custom4/0.1.4

That is being found via::

    N[blyth@localhost qudarap]$ echo $CMAKE_PREFIX_PATH | tr ":" "\n" | grep custom4
    /data/blyth/junotop/ExternalLibs/custom4/0.1.4



::

    098 function juno-ext-libs-custom4-install-dir {
     99     local version=${1:-$(juno-ext-libs-custom4-version)}
    100     echo $(juno-ext-libs-install-root)/$(juno-ext-libs-custom4-name)/$version
    101 }



c4/build.sh
-------------

Direct install into OPTICKS_PREFIX is far too dirty::

    epsilon:customgeant4 blyth$ ./build.sh info
                              sdir : /Users/blyth/customgeant4 
                              name : customgeant4 
                              BASE : /tmp/blyth/customgeant4 
                              bdir : /tmp/blyth/customgeant4/build 
                              idir : /usr/local/opticks 
                               arg : info 
                    OPTICKS_PREFIX : /usr/local/opticks 
    epsilon:customgeant4 blyth$ 


::

    epsilon:customgeant4 blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:customgeant4 blyth$ 


These opticks/externals are ones that Opticks manages the install of::

    epsilon:customgeant4 blyth$ l /usr/local/opticks/externals/
    total 8
    0 drwxr-xr-x  39 blyth  staff  1248 Nov 12  2022 ..
    0 drwxr-xr-x  17 blyth  staff   544 Oct  6  2022 plog
    0 drwxr-xr-x  25 blyth  staff   800 Oct  6  2022 .
    0 drwxr-xr-x   7 blyth  staff   224 Jul  6  2021 glm
    0 drwxr-xr-x  11 blyth  staff   352 Feb 26  2021 plog.old
    0 drwxr-xr-x  12 blyth  staff   384 Feb 10  2021 owl
    0 drwxr-xr-x  12 blyth  staff   384 Dec  4  2020 include
    0 drwxr-xr-x   4 blyth  staff   128 Dec  3  2020 optix7c
    0 drwxr-xr-x  28 blyth  staff   896 Sep 16  2020 lib
    0 drwxr-xr-x   8 blyth  staff   256 Sep 16  2020 openmesh
    0 -rw-r--r--   1 blyth  staff     0 Jun  6  2020 opticks-setup-generate
    0 drwxr-xr-x   3 blyth  staff    96 May 11  2020 bin
    0 drwxr-xr-x   3 blyth  staff    96 May 11  2020 share
    0 drwxr-xr-x   4 blyth  staff   128 May  7  2020 DualContouringSample
    8 -rw-r--r--   1 blyth  staff  2058 May  5  2020 opticks-envg4.bash
    0 drwxr-xr-x   4 blyth  staff   128 May  5  2020 ImplicitMesher
    0 drwxr-xr-x   4 blyth  staff   128 May  4  2020 assimp
    0 drwxr-xr-x   8 blyth  staff   256 May  4  2020 glfw
    0 drwxr-xr-x   4 blyth  staff   128 Apr  9  2020 yoctogl
    0 drwxr-xr-x   6 blyth  staff   192 Oct 20  2018 gleq
    0 drwxr-xr-x   4 blyth  staff   128 Jul  7  2018 imgui
    0 drwxr-xr-x   4 blyth  staff   128 May 19  2018 csgbsp
    0 drwxr-xr-x   4 blyth  staff   128 May 17  2018 bcm
    0 lrwxr-xr-x   1 blyth  staff    21 Apr  4  2018 libassimp.3.dylib -> lib/libassimp.3.dylib
    0 drwxr-xr-x   4 blyth  staff   128 Apr  4  2018 glew
    epsilon:customgeant4 blyth$ 


Conversely the opticks_externals are ones expected to be installed by other means.::

    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
 
So it makes more sense to install into::

    /usr/local/opticks_externals/custom4/




DONE : manually cleaned the below
------------------------------------


::

    epsilon:opticks blyth$ pwd
    /usr/local/opticks
    epsilon:opticks blyth$ find . -name '*Custom4*' 
    ./include/Custom4
    ./lib/libCustom4.dylib
    ./lib/Custom4-0.1.4
    ./lib/Custom4-0.1.4/Custom4Config.cmake
    ./lib/Custom4-0.1.4/Custom4ConfigVersion.cmake
    ./lib/Custom4-0.1.5
    ./lib/Custom4-0.1.5/Custom4Config.cmake
    ./lib/Custom4-0.1.5/Custom4ConfigVersion.cmake
    ./lib/UseCustom4
    ./lib/U4Custom4Test
    ./build/u4/tests/CMakeFiles/U4Custom4Test.dir
    ./build/u4/tests/CMakeFiles/U4Custom4Test.dir/U4Custom4Test.cc.o
    ./build/u4/tests/U4Custom4Test
    epsilon:opticks blyth$ 


DONE : changed c4/build.sh to install into versioned dirs
-----------------------------------------------------------

::

    -- Install configuration: "Debug"
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/lib/libCustom4.dylib
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4OpBoundaryProcess.hh
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4IPMTAccessor.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4CustomART.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4CustomART_Debug.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4MultiLayrStack.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4Touchable.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4TrackInfo.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4Track.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4Pho.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4GS.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/C4Sys.h
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5/Custom4Config.cmake
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5/Custom4ConfigVersion.cmake
    -- Installing: /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5/C4Version.h


WIP : get qu to find c4, by changing laptop $HOME/.opticks_config 
--------------------------------------------------------------------

As expected need to change CMAKE_PREFIX_PATH to find the versioned c4. 

::

    epsilon:qudarap blyth$ om-clean
    rm -rf /usr/local/opticks/build/qudarap && mkdir -p /usr/local/opticks/build/qudarap
    epsilon:qudarap blyth$ om-conf


$HOME/.opticks_config the appropriate place to change CMAKE_PREFIX_PATH is 
just after Geant4::

     47 # PATH envvars control the externals that opticks/CMake or pkg-config will find  
     48 unset CMAKE_PREFIX_PATH
     49 unset PKG_CONFIG_PATH
     50 
     51 # mandatory envvars in buildenv 
     52 
     53 # OPTICKS_PREFIX is overriden in the below sourcing of opticks_setup
     54 #export OPTICKS_PREFIX=/usr/local/opticks
     55 #export OPTICKS_PREFIX=/usr/local/opticks_min
     56 
     57 export OPTICKS_CUDA_PREFIX=/usr/local/cuda
     58 export OPTICKS_OPTIX_PREFIX=/usr/local/optix
     59 export OPTICKS_COMPUTE_CAPABILITY=30
     60 
     61 export OPTICKS_OPTIX5_PREFIX=/usr/local/optix
     62 export OPTICKS_OPTIX7_PREFIX=/Developer/OptiX_700
     63 
     64 
     65 ## hookup paths to access "foreign" externals 
     66 opticks-prepend-prefix /usr/local/opticks_externals/boost
     67 opticks-prepend-prefix /usr/local/opticks_externals/xercesc
     68 
     69 # leave only one of the below clhep+geant4 setup "stanzas" uncommented 
     70 # to pick the geant4 version and start a new session before doing anything 
     71 # like using the g4- functions or building opticks against this geant4 
     72 
     73 # standard 1042 
     74 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     75 opticks-prepend-prefix /usr/local/opticks_externals/g4_1042

::

     74 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     75 opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
     76 opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.5


I dont think the ordering is very important::

    epsilon:customgeant4 blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4/0.1.5
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:customgeant4 blyth$ 


qu finds it and succeeds to build against c4 0.1.5 

::

    -- PACKAGE_VERSION_UNSUITABLE  :  
    -- Custom4_VERBOSE       : ON 
    -- Custom4_FOUND         : YES 
    -- Custom4_VERSION       : 0.1.5 
    -- Custom4_PREFIX        : /usr/local/opticks_externals/custom4/0.1.5 
    -- Custom4_INCLUDE_DIR   : /usr/local/opticks_externals/custom4/0.1.5/include/Custom4 
    -- Custom4_INCLUDE_DIRS  : /usr/local/opticks_externals/custom4/0.1.5/include/Custom4 
    -- Custom4_CFLAGS        : -I/usr/local/opticks_externals/custom4/0.1.5/include/Custom4 
    -- Custom4_DEFINITIONS   : -DWITH_CUSTOM4 
    -- Custom4_LIBRARY_DIR   : /usr/local/opticks_externals/custom4/0.1.5/lib 
    -- Custom4_LIBRARY_PATH  : /usr/local/opticks_externals/custom4/0.1.5/lib/libCustom4.dylib 
    -- Custom4_LIBRARIES     : -L/usr/local/opticks_externals/custom4/0.1.5/lib -lCustom4 
    -- Custom4_CMAKE_PATH    : /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5/Custom4Config.cmake 
    -- Custom4_CMAKE_DIR     : /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5 
    -- Configuring done


BUT, Note that C4Version.h is ending up in wrong dir::

    epsilon:qudarap blyth$ cat /usr/local/opticks_externals/custom4/0.1.5/lib/Custom4-0.1.5/C4Version.h
    #pragma once

    #define Custom4_VERSION_MAJOR 0
    #define Custom4_VERSION_MINOR 1
    #define Custom4_VERSION_PATCH 5
    #define Custom4_VERSION 0.1.5

It needs to be together with the other headers::

    epsilon:qudarap blyth$ l /usr/local/opticks_externals/custom4/0.1.5/include/Custom4/
    total 192
     0 drwxr-xr-x  13 blyth  staff    416 Jul  2 12:26 .
     0 drwxr-xr-x   3 blyth  staff     96 Jul  2 12:26 ..
    48 -rw-r--r--   1 blyth  staff  23517 Jun 25 14:51 C4MultiLayrStack.h
    32 -rw-r--r--   1 blyth  staff  12430 Jun 24 19:13 C4CustomART.h
     8 -rw-r--r--   1 blyth  staff   2079 Apr  7 15:21 C4GS.h
    16 -rw-r--r--   1 blyth  staff   5483 Apr  7 15:02 C4Track.h
    16 -rw-r--r--   1 blyth  staff   6105 Apr  7 11:44 C4Pho.h
     8 -rw-r--r--   1 blyth  staff    718 Mar 29 19:59 C4CustomART_Debug.h
     8 -rw-r--r--   1 blyth  staff   3831 Mar 24 12:45 C4TrackInfo.h
    24 -rw-r--r--   1 blyth  staff  12262 Mar 24 12:43 C4OpBoundaryProcess.hh
     8 -rw-r--r--   1 blyth  staff    442 Mar 24 12:39 C4Sys.h
    16 -rw-r--r--   1 blyth  staff   5415 Mar 24 12:05 C4Touchable.h
     8 -rw-r--r--   1 blyth  staff    453 Mar 22 11:03 C4IPMTAccessor.h
    epsilon:qudarap blyth$ 

Changed the destination::

    191 install(FILES
    192   ${PROJECT_BINARY_DIR}/Custom4Config.cmake
    193   ${PROJECT_BINARY_DIR}/Custom4ConfigVersion.cmake
    194   DESTINATION ${CUSTOM4_CMAKE_DIR}
    195   )
    196 
    197 install(FILES
    198   ${PROJECT_BINARY_DIR}/C4Version.h
    199   DESTINATION ${CUSTOM4_RELATIVE_INCDIR}
    200   )



c4 return to v0.1.4
----------------------


::

    epsilon:customgeant4 blyth$ git checkout tags/v0.1.4 
    Note: checking out 'tags/v0.1.4'.

    You are in 'detached HEAD' state. You can look around, make experimental
    changes and commit them, and you can discard any commits you make in this
    state without impacting any branches by performing another checkout.

    If you want to create a new branch to retain commits you create, you may
    do so (now or later) by using -b with the checkout command again. Example:

      git checkout -b <new-branch-name>

    HEAD is now at 33fc856... bump to 0.1.4
    epsilon:customgeant4 blyth$ 



Install that into a versioned dir::

    OPTICKS_PREFIX=/usr/local/opticks_externals/custom4/0.1.4 ./build.sh 

Actually that didnt work because login scripts depend on OPTICKS_PREFIX, had to change the build.sh to accept CUSTOM4_PREFIX::

    CUSTOM4_PREFIX=/usr/local/opticks_externals/custom4/0.1.4 ./build.sh 

Also kludge up the C4Version.h::

    epsilon:qudarap blyth$ cat /usr/local/opticks_externals/custom4/0.1.4/include/C4Version.h
    cat: /usr/local/opticks_externals/custom4/0.1.4/include/C4Version.h: No such file or directory
    epsilon:qudarap blyth$ echo "#define Custom4_VERSION_NUMBER 00104" > /usr/local/opticks_externals/custom4/0.1.4/include/C4Version.h
    epsilon:qudarap blyth$ cat /usr/local/opticks_externals/custom4/0.1.4/include/C4Version.h
    #define Custom4_VERSION_NUMBER 00104
    epsilon:qudarap blyth$ 

These change .opticks_config to use the old c4::

     73 # standard 1042 
     74 opticks-prepend-prefix /usr/local/opticks_externals/clhep
     75 opticks-prepend-prefix /usr/local/opticks_externals/g4_1042
     76 #opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.5
     77 opticks-prepend-prefix /usr/local/opticks_externals/custom4/0.1.4
     78 

Start new shell and check CMAKE_PREFIX_PATH::

    epsilon:qudarap blyth$ echo $CMAKE_PREFIX_PATH | tr ":" "\n"
    /usr/local/opticks_externals/custom4/0.1.4
    /usr/local/opticks_externals/g4_1042
    /usr/local/opticks_externals/clhep
    /usr/local/opticks_externals/xercesc
    /usr/local/opticks_externals/boost
    /usr/local/opticks
    /usr/local/opticks/externals
    /usr/local/optix
    epsilon:qudarap blyth$ 

Try rebuild qu against 0.1.4::

    qu
    om-
    om-clean
    om-conf   # it finds 0.1.4
    om


That fails in the expected way::

    [  6%] Building NVCC (Device) object CMakeFiles/QUDARap.dir/QUDARap_generated_QEvent.cu.o
    /Users/blyth/opticks/qudarap/qpmt.h(184): error: no instance of constructor "Stack<T, N>::Stack [with T=float, N=4]" matches the argument list
                argument types are: (float, float, float, float [16], unsigned int)
              detected during:
                instantiation of "void qpmt<F>::get_lpmtid_LL(F *, int, F, F, F) const [with F=float]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(244): here
                instantiation of "void _QPMT_mct_lpmtid<F,P>(qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float, P=16]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(282): here
                instantiation of "void QPMT_mct_lpmtid(dim3, dim3, qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(302): here

    /Users/blyth/opticks/qudarap/qpmt.h(185): error: class "Layr<float>" has no member "cdata"
              detected during:
                instantiation of "void qpmt<F>::get_lpmtid_LL(F *, int, F, F, F) const [with F=float]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(244): here
                instantiation of "void _QPMT_mct_lpmtid<F,P>(qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float, P=16]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(282): here
                instantiation of "void QPMT_mct_lpmtid(dim3, dim3, qpmt<F> *, int, F *, const F *, unsigned int, const int *, unsigned int) [with F=float]" 
    /Users/blyth/opticks/qudarap/QPMT.cu(302): here





