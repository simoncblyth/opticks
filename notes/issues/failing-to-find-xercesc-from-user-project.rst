failing-to-find-xercesc-from-user-project
===========================================


issue
----------

* this fail doesnt happen on workstation, probably as xercesc is from system there 

::

    [blyth@lxslc701 CerenkovMinimal]$ ./go-release.sh 
    pfx /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg
    ...
    CMake Error at /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package):
      Found package configuration file:

        /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib64/cmake/extg4/extg4-config.cmake

      but it set ExtG4_FOUND to FALSE so package "ExtG4" is considered to be NOT
      FOUND.  Reason given by package:

      ExtG4 could not be found because dependency OpticksXercesC could not be
      found.

    Call Stack (most recent call first):
      /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib64/cmake/cfg4/cfg4-config.cmake:9 (find_dependency)
      /usr/share/cmake3/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib64/cmake/g4ok/g4ok-config.cmake:7 (find_dependency)
      CMakeLists.txt:11 (find_package)

    -- Configuring incomplete, errors occurred!
    See also "/tmp/blyth/opticks/examples/CerenkovMinimal/build/CMakeFiles/CMakeOutput.log".
    /tmp/blyth/opticks/examples/CerenkovMinimal/build
    make: *** No targets specified and no makefile found.  Stop.
    [blyth@lxslc701 CerenkovMinimal]$ 



* imported dependency tree from G4OK incomplete ?

CMakeLists.txt::

 01 cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
  2 set(name CerenkovMinimal)
  3 project(${name} VERSION 0.1.0)
  4 include(OpticksBuildOptions)
  5 
  6 if(POLICY CMP0077)  # see note/issues/cmake-3.13.4-FindCUDA-warnings.rst
  7     cmake_policy(SET CMP0077 OLD)
  8 endif()
  9 
 10 
 11 find_package( G4OK CONFIG REQUIRED )
 12 


cmake/g4ok/g4ok-config.cmake::

     01 
      2 # TOPMATTER
      3 
      4 
      5 include(CMakeFindDependencyMacro)
      6 # Library: Opticks::CFG4
      7 find_dependency(CFG4)
      8 # Library: Opticks::ExtG4
      9 find_dependency(ExtG4)
     10 # Library: Opticks::OKOP
     11 find_dependency(OKOP)
     12 
     13 include("${CMAKE_CURRENT_LIST_DIR}/g4ok-targets.cmake")
     14 include("${CMAKE_CURRENT_LIST_DIR}/properties-g4ok-targets.cmake")


cmake/cfg4/cfg4-config.cmake::

     01 
      2 # TOPMATTER
      3 
      4 
      5 include(CMakeFindDependencyMacro)
      6 # Library: Opticks::G4
      7 find_dependency(G4 MODULE REQUIRED)
      8 # Library: Opticks::ExtG4
      9 find_dependency(ExtG4)
     10 # Library: Opticks::OpticksXercesC
     11 find_dependency(OpticksXercesC)
     12 # Library: Opticks::OpticksGeo
     13 find_dependency(OpticksGeo)
     14 # Library: Opticks::ThrustRap
     15 find_dependency(ThrustRap)
     16 
     17 include("${CMAKE_CURRENT_LIST_DIR}/cfg4-targets.cmake")
     18 include("${CMAKE_CURRENT_LIST_DIR}/properties-cfg4-targets.cmake")


cmake/extg4/extg4-config.cmake::

     01 
      2 # TOPMATTER
      3 
      4 
      5 include(CMakeFindDependencyMacro)
      6 # Library: Opticks::G4
      7 find_dependency(G4 MODULE REQUIRED)
      8 # Library: Opticks::GGeo
      9 find_dependency(GGeo)
     10 # Library: Opticks::OpticksXercesC
     11 find_dependency(OpticksXercesC)
     12 
     13 include("${CMAKE_CURRENT_LIST_DIR}/extg4-targets.cmake")
     14 include("${CMAKE_CURRENT_LIST_DIR}/properties-extg4-targets.cmake")




Minimal Reproducer
---------------------

Added examples/UseOpticksXercesC/go-release.sh::

     23 pfx=$(opticks-release-prefix)
     24 
     25 sdir=$(pwd)
     26 bdir=/tmp/$USER/opticks/$(basename $sdir)/build
     27 idir=$HOME
     28 
     29 rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd
     30 
     31 
     32 cmake3 $sdir \
     33     -DCMAKE_BUILD_TYPE=Debug \
     34     -DCMAKE_PREFIX_PATH="$pfx/externals;$pfx" \
     35     -DCMAKE_MODULE_PATH=$pfx/cmake/Modules \
     36     -DCMAKE_INSTALL_PREFIX=$idir
     37 
     38 
     39 make
     40 make install
        

The lib was found, but not the include dir::

    [blyth@lxslc701 UseOpticksXercesC]$ ./go-release.sh 
    ...
    -- Configuring UseOpticksXercesC
    --  Below two strings differ : forced to use absolute RPATH 
    --  CMAKE_INSTALL_PREFIX : /afs/ihep.ac.cn/users/b/blyth 
    --  OPTICKS_PREFIX       : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg 
    -- FindOpticksXercesC.cmake OpticksXercesC_MODULE      : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindOpticksXercesC.cmake  
    -- FindOpticksXercesC.cmake OpticksXercesC_INCLUDE_DIR : OpticksXercesC_INCLUDE_DIR-NOTFOUND  
    -- FindOpticksXercesC.cmake OpticksXercesC_LIBRARY     : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/lib/libxerces-c.so  
    -- FindOpticksXercesC.cmake OpticksXercesC_FOUND       : NO  
    -- OpticksXercesC_MODULE  : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindOpticksXercesC.cmake 
    -- Configuring done
    CMake Error at CMakeLists.txt:11 (add_executable):
      Target "UseOpticksXercesC" links to target "Opticks::OpticksXercesC" but
      the target was not found.  Perhaps a find_package() call is missing for an
      IMPORTED target, or an ALIAS target is missing?


    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks/UseOpticksXercesC/build
    Scanning dependencies of target UseOpticksXercesC
    [ 50%] Building CXX object CMakeFiles/UseOpticksXercesC.dir/UseOpticksXercesC.cc.o
    /hpcfs/juno/junogpu/blyth/opticks/examples/UseOpticksXercesC/UseOpticksXercesC.cc:22:31: fatal error: xercesc/dom/DOM.hpp: No such file or directory
     #include <xercesc/dom/DOM.hpp>
                                   ^
    compilation terminated.
    make[2]: *** [CMakeFiles/UseOpticksXercesC.dir/UseOpticksXercesC.cc.o] Error 1
    make[1]: *** [CMakeFiles/UseOpticksXercesC.dir/all] Error 2
    make: *** [all] Error 2
    [ 50%] Building CXX object CMakeFiles/UseOpticksXercesC.dir/UseOpticksXercesC.cc.o
    /hpcfs/juno/junogpu/blyth/opticks/examples/UseOpticksXercesC/UseOpticksXercesC.cc:22:31: fatal error: xercesc/dom/DOM.hpp: No such file or directory
     #include <xercesc/dom/DOM.hpp>
                                   ^
    compilation terminated.
    make[2]: *** [CMakeFiles/UseOpticksXercesC.dir/UseOpticksXercesC.cc.o] Error 1
    make[1]: *** [CMakeFiles/UseOpticksXercesC.dir/all] Error 2
    make: *** [all] Error 2
    [blyth@lxslc701 UseOpticksXercesC]$ 
        


The includes are indeed missing from the installed externals:: 

    [blyth@lxslc701 ~]$ l /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/include/
    total 16
    drwxr-xr-x 3 blyth dyw 4096 Sep 29 21:44 assimp
    drwxr-xr-x 2 blyth dyw 4096 Sep 29 21:44 DualContouringSample
    drwxr-xr-x 4 blyth dyw 4096 Sep 29 21:44 OpenMesh
    drwxr-xr-x 3 blyth dyw 4096 Sep 29 21:44 YoctoGL
    [blyth@lxslc701 ~]$ 


Contrast with the source build, opticks-cd::

    [blyth@lxslc701 opticks]$ l externals/include/
    total 284
    drwxr-xr-x  2 blyth dyw   4096 Sep 26 16:53 GL
    drwxr-xr-x  4 blyth dyw 245760 Apr 28 17:00 Geant4
    drwxr-xr-x 11 blyth dyw   4096 Apr 28 14:57 xercesc
    drwxr-xr-x  2 blyth dyw   4096 Apr 28 14:51 CSGBSP
    drwxr-xr-x  3 blyth dyw   4096 Apr 28 14:50 YoctoGL
    drwxr-xr-x  2 blyth dyw   4096 Apr 28 14:50 DualContouringSample
    drwxr-xr-x  2 blyth dyw   4096 Apr 28 14:49 ImplicitMesher
    drwxr-xr-x  4 blyth dyw   4096 Apr 28 14:45 OpenMesh
    drwxr-xr-x  3 blyth dyw   4096 Apr 28 14:40 assimp
    drwxr-xr-x  2 blyth dyw   4096 Apr 28 14:36 ImGui
    drwxr-xr-x  2 blyth dyw   4096 Apr 28 14:34 GLFW




Inappropriate fix : just include xercesc with okdist
--------------------------------------------------------------------

* add to bin/okdist.py
* hmm adding xercesc includes and libs (which were already there) to the okdist distribution is the easy fix 

  * but not appropriate given it being a geant4 dependency and are not including geant4 
  * need to treat xercesc the same as non-included geant4
 

::

    [blyth@lxslc701 UseOpticksXercesC]$ ./go-release.sh 
    /tmp/blyth/opticks/UseOpticksXercesC/build
    ...
    -- Configuring UseOpticksXercesC
    ...
    -- FindOpticksXercesC.cmake OpticksXercesC_MODULE      : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindOpticksXercesC.cmake  
    -- FindOpticksXercesC.cmake OpticksXercesC_INCLUDE_DIR : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/include  
    -- FindOpticksXercesC.cmake OpticksXercesC_LIBRARY     : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/lib/libxerces-c.so  
    -- FindOpticksXercesC.cmake OpticksXercesC_FOUND       : YES  
    -- OpticksXercesC_MODULE  : /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/cmake/Modules/FindOpticksXercesC.cmake 
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /tmp/blyth/opticks/UseOpticksXercesC/build
    Scanning dependencies of target UseOpticksXercesC
    ...
    [100%] Linking CXX executable UseOpticksXercesC
    [100%] Built target UseOpticksXercesC
    [100%] Built target UseOpticksXercesC
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/lib/UseOpticksXercesC
    -- Set runtime path of "/afs/ihep.ac.cn/users/b/blyth/lib/UseOpticksXercesC" to "/hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/lib64:/hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/lib:/hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/lib64:/hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg/externals/OptiX/lib64"
    [blyth@lxslc701 UseOpticksXercesC]$ 



::

    [blyth@lxslc701 CerenkovMinimal]$ ./go-release.sh 
    pfx /hpcfs/juno/junogpu/blyth/local/opticks/releases/Opticks-0.0.0_alpha/x86_64-slc7-gcc48-geant4_10_04_p02-dbg
    -- The C compiler identification is GNU 4.8.5
    ...
    Install the project...
    -- Install configuration: "Debug"
    -- Installing: /afs/ihep.ac.cn/users/b/blyth/lib/CerenkovMinimal



Executable runs until attempting to use GPU::

    2019-10-05 11:13:26.368 ERROR [24862] [OContext::SetupOptiXCachePathEnvvar@284] envvar OPTIX_CACHE_PATH not defined setting it internally to /var/tmp/blyth/OptiXCache
    2019-10-05 11:13:26.424 INFO  [24862] [OContext::InitRTX@321]  --rtx 0 setting  OFF
    terminate called after throwing an instance of 'APIError'
    Aborted (core dumped)



Need to add xercesc external lib and include handling to opticks-envg4 and g4- 

* opticks-envg4 is kinda a mockup of an externally managed geant4, 
  so perhaps more realistic to not use BCM and imported targets ?

* geant4 dependency is treated in cmake/Modules/FindG4.cmake as an imported target

  * am not going to get rid of that, so try to just add xercesc to this

/home/blyth/local/opticks/externals/g4/geant4.10.04.p02/examples/extended/persistency/gdml/G01/README::

   34 - You need to have built the persistency/gdml module by having
   35   set the -DGEANT4_USE_GDML=ON flag during the CMAKE configuration step,
   36   as well as the -DXERCESC_ROOT_DIR=<path_to_xercesc> flag pointing to
   37   the path where the XercesC XML parser package is installed in your system.
   38 




::

     05 opticks-envg4-source(){ echo $BASH_SOURCE ; }
      6 opticks-envg4-vi(){  vi $BASH_SOURCE ; }
      7 opticks-envg4-dir(){ echo $(dirname $BASH_SOURCE) ; }
      8 
      9 opticks-envg4-name(){ echo Geant4-10.4.2 ; }
     10 opticks-envg4-Geant4_DIR(){ echo $(opticks-envg4-dir)/lib64/$(opticks-envg4-name) ; }
     11 
     12 opticks-envg4-info(){ cat << EOI
     13 
     14     opticks-envg4-source      : $(opticks-envg4-source)
     15     opticks-envg4-dir         : $(opticks-envg4-dir)
     16     opticks-envg4-name        : $(opticks-envg4-name)
     17     opticks-envg4-Geant4_DIR  : $(opticks-envg4-Geant4_DIR)
     18 
     19 EOI
     20 }
     21 
     22 opticks-envg4-main(){
     23 
     24     local here=$(opticks-envg4-dir)
     25 
     26 
     27     export G4NEUTRONHPDATA=$here/share/Geant4-10.4.2/data/G4NDL4.5
     28     export G4PIIDATA=$here/share/Geant4-10.4.2/data/G4PII1.3
     29     export G4NEUTRONXSDATA=$here/share/Geant4-10.4.2/data/G4NEUTRONXS1.4
     30     export G4LEDATA=$here/share/Geant4-10.4.2/data/G4EMLOW7.3
     31     export G4REALSURFACEDATA=$here/share/Geant4-10.4.2/data/RealSurface2.1.1
     32     export G4ENSDFSTATEDATA=$here/share/Geant4-10.4.2/data/G4ENSDFSTATE2.2
     33     export G4ABLADATA=$here/share/Geant4-10.4.2/data/G4ABLA3.1
     34     export G4RADIOACTIVEDATA=$here/share/Geant4-10.4.2/data/RadioactiveDecay5.2
     35     export G4LEVELGAMMADATA=$here/share/Geant4-10.4.2/data/PhotonEvaporation5.2
     36     export G4SAIDXSDATA=$here/share/Geant4-10.4.2/data/G4SAIDDATA1.1
     37 
     38     export LD_LIBRARY_PATH=$here/lib64:$LD_LIBRARY_PATH
     39 
     40 }
     41 
     42 opticks-envg4-main


 
Hmm, would be more appropriate for Geant4 to not be in opticks/externals::

    [blyth@lxslc701 CerenkovMinimal]$ opticks-envg4-Geant4_DIR
    /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib64/Geant4-10.4.2



g4-cmake
--------------

::

    528 g4-cmake(){
    529    local iwd=$PWD
    530 
    531    local bdir=$(g4-bdir)
    532    mkdir -p $bdir
    533 
    534    local idir=$(g4-prefix)
    535    mkdir -p $idir
    536 
    537    g4-cmake-info
    538 
    539    g4-bcd
    540 
    541    cmake \
    542        -G "$(opticks-cmake-generator)" \
    543        -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
    544        -DGEANT4_INSTALL_DATA=ON \
    545        -DGEANT4_USE_GDML=ON \
    546        -DXERCESC_LIBRARY=$(xercesc-library) \
    547        -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
    548        -DCMAKE_INSTALL_PREFIX=$idir \
    549        $(g4-dir)
    550 
    551    cd $iwd
    552 }


::

    [blyth@lxslc701 Modules]$ xercesc-info

    xercesc-info
    ==============

    USED BY CMAKE, FOR EITHER SYSTEM OR MANULLY INSTALLED XERCES-C

       xercesc-library : /hpcfs/juno/junogpu/blyth/local/opticks/externals/lib/libxerces-c-3.1.so
       xercesc-include-dir : /hpcfs/juno/junogpu/blyth/local/opticks/externals/include

    ONLY RELEVANT WHEN BUILDING MANUALLY 

       xercesc-url    : http://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.1.1.tar.gz
       xercesc-dist   : xerces-c-3.1.1.tar.gz
       xercesc-name   : xerces-c-3.1.1
       xercesc-base   : /hpcfs/juno/junogpu/blyth/local/opticks/externals/xercesc
       xercesc-dir    : /hpcfs/juno/junogpu/blyth/local/opticks/externals/xercesc/xerces-c-3.1.1
       xercesc-bdir   : /hpcfs/juno/junogpu/blyth/local/opticks/externals/xercesc/xerces-c-3.1.1.build

       xercesc-prefix  : /hpcfs/juno/junogpu/blyth/local/opticks/externals






