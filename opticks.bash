opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }

opticks-devnotes(){ cat << EOU

Opticks Dev Notes
====================

Aiming for opticks.bash to go in top level of a new Opticks repo
together with top level superbuild CMakeLists.txt
Intend to allow building independent of the env.


Windows 7 MSYS2 build
-----------------------

Steps::
  
   # installed MSYS2 following web instructions to setup/update pacman

   pacman -S git
   pacman -S mercurial

   hg clone http://bitbucket.org/simoncblyth/env

   # hookup env to .bash_profile 
   # alias vi="vim"

   pacman -S cmake

   opticks-;opticks-externals-install   

   # glm- misses unzip
   pacman -S unzip

   # imgui- misses diff
   pacman -Ss diff
   pacman -S diffutils

   # assimp-get : connection reset by peer, but switch from git: to http: protocol and it works
   # openmesh-get : misses tar

   pacman -S tar 

   # glfw-cmake : runs into lack of toolchain

   pacman -S mingw-w64-x86_64-cmake    # from my notes msys2-

   # glfw-cmake : misses make

   pacman -S mingw-w64-x86_64-make     # guess
   pacman -S mingw-w64-x86_64-toolchain     # guess
   pacman -S base-devel           # guess
  
   # how to setup path to use the right ones ? 
   # see .bash_profile putting /mingw64/bin at head

   # adjust glfw-cmake imgui-cmake to specify opticks-cmake-generator 
   # glfw--   succeeds to cmake/make
   # glew--   succeeds to make (no cmake)
   # imgui-cmake   problems finding GLEW   
   #     maybe windows directory issue 
   #        what in MSYS2 appears as /usr/local/opticks/externals/
   #        is actually at C:\msys64\usr\local\opticks\externals\ 
   #     nope looks like windows lib naming, on mac its libGLEW.dylib on windows libglew32.a and libglew32.dll.a
   #     problem seems to be glew-get use of symbolic link, change glew-idir to avoid enables the find
   # imgui-cmake succeed
   # imgui-make  : undefined references in link to glGetIntegerv etc...
   #     
   # imgui-- : needed to configure the opengl libs and set defintion to avoid IME 
   #
   # assimp-cmake : fails to find DirectX : avoid by switch off tool building
   # opticks-cmake : fails to find Boost 

   pacman -S mingw-w64-x86_64-boost

    


See Also
----------

cmake-
    background on cmake

cmakex-
    documenting the development of the opticks- cmake machinery 

cmakecheck-
    testing CMake config

Fullbuild Testing
------------------

Only needed whilst making sweeping changes::

    simon:~ blyth$ opticks-distclean         # check what will be deleted
    simon:~ blyth$ opticks-distclean | sh    # delete 

    simon:~ blyth$ opticks-fullclean         # check what will be deleted
    simon:~ blyth$ opticks-fullclean | sh    # delete 

    simon:~ blyth$ opticks- ; opticks-full


G4PB build
-----------

* /usr/local/opticks : permission denied


glfw 3.1.1 needs OSX 10.6+ ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* TODO: examine glfw history to find older version prior to 10.6 isms


G5 build
----------

* glfw:CMake 2.8.9 or higher is required.  You are running version 2.6.4


runtime path problem
~~~~~~~~~~~~~~~~~~~~~~

::

    -- Up-to-date: /home/blyth/local/opticks/gl/tex/vert.glsl
    -- Installing: /home/blyth/local/opticks/lib/libGGeoViewLib.so
    -- Set runtime path of "/home/blyth/local/opticks/lib/libGGeoViewLib.so" to ""
    -- Installing: /home/blyth/local/opticks/bin/GGeoView
    -- Set runtime path of "/home/blyth/local/opticks/bin/GGeoView" to ""
    -- Up-to-date: /home/blyth/local/opticks/include/GGeoView/App.hh

* currently kludged via LD_LIBRARY_PATH


NPY/jsonutil.cpp boost::ptree compilation warning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Maybe split off ptree dependency togther with BRegex or BCfg


remote running : X11 failed to connect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you do not ssh in with -Y or X11 forwarding has been disabled you will get::

    X11: Failed to open X display


X11 : headless end of the line ?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

    simon:env blyth$ ssh -Y G5
    Enter passphrase for key '/Users/blyth/.ssh/id_rsa': 
    Last login: Wed Apr 27 14:17:19 2016 from simon.phys.ntu.edu.tw
    [blyth@ntugrid5 ~]$ opticks-
    [blyth@ntugrid5 ~]$ opticks-check
    [2016-04-27 18:02:59.379580] [0x00002b55c3eaafe0] [info]    Opticks::preargs argc 2 argv[0] /home/blyth/local/opticks/bin/GGeoView mode Interop
    ...
    [2016-04-27 18:02:59.389754]:info: App::prepareViz size 2880,1704,2,0 position 200,200,0,0
    X11: RandR gamma ramp support seems brokenGLX: Failed to create context: GLXBadFBConfig
    [blyth@ntugrid5 ~]$ 


* reducing size to 1024,768,1 made no difference
* also running from an X11 terminal made no difference

* http://inviwo.org/svn/inviwo/modules/glfw/ext/glfw/src/x11_gamma.c


Locating Boost, CUDA, OptiX
------------------------------

CMake itself provides cross platform machinery to find Boost and CUDA::

   /opt/local/share/cmake-3.4/Modules/FindBoost.cmake
   /opt/local/share/cmake-3.4/Modules/FindCUDA.cmake 

OptiX provides eg::

   /Developer/OptiX_380/SDK/CMake/FindOptiX.cmake

Tis self contained so copy into cmake/Modules
to avoid having to set CMAKE_MODULE_PATH to find it.  
This provides cache variable OptiX_INSTALL_DIR.



TODO
-----

* find out what depends on ssl and crypt : maybe in NPY_LIBRARIES 
* tidy up optix optixu FindOptiX from the SDK doesnt set OPTIX_LIBRARIES

* get the CTest tests to pass 

* incorporate cfg4- in superbuild with G4 checking

* check OptiX 4.0 beta for cmake changes 
* externalize or somehow exclude from standard building the Rap pkgs, as fairly stable
* look into isolating Assimp dependency usage

* spawn Opticks repository 
* adopt single level directories 
* split ggv- usage from ggeoview- building
* rename GGeoView to OpticksView/OpticksViz
* rename GGeo to OpticksGeo ?

* investigate CPack as way of distributing binaries



TODO: are the CUDA flags being used
------------------------------------

::

    simon:env blyth$ optix-cuda-nvcc-flags
    -ccbin /usr/bin/clang --use_fast_math


TODO: make envvar usage optional
----------------------------------

Enable all envvar settings to come in via the metadata .ini approach with 
envvars being used to optionally override those.

Bash launcher ggv.sh tied into the individual bash functions and 
sets up envvars::

   OPTICKS_GEOKEY
   OPTICKS_QUERY
   OPTICKS_CTRL
   OPTICKS_MESHFIX
   OPTICKS_MESHFIX_CFG

TODO:cleaner curand state
---------------------------

File level interaction between optixrap- and cudarap- 
in order to persist the state currently communicates via envvar ?

:: 

    simon:~ blyth$ l /usr/local/env/graphics/ggeoview/cache/rng
    total 344640
    -rw-r--r--  1 blyth  staff   44000000 Dec 29 20:33 cuRANDWrapper_1000000_0_0.bin
    -rw-r--r--  1 blyth  staff     450560 May 17  2015 cuRANDWrapper_10240_0_0.bin
    -rw-r--r--  1 blyth  staff  132000000 May 17  2015 cuRANDWrapper_3000000_0_0.bin


EOU
}

opticks-usage(){ cat << EOU

Opticks
=========

Get Opticks 
-----------

Clone the repository from bitbucket::

   cd
   hg clone http://bitbucket.org/simoncblyth/opticks 

Connect the opticks bash functions to your shell by adding a line to your .bash_profile::

   opticks-(){ . $HOME/opticks/opticks.bash && opticks-env $* ; }

This defines the bash function *opticks-* that is termed a precursor function 
as running it will define other functions all starting with *opticks-* such as *opticks-vi*
and *opticks-usage*.

Build Tools
------------

Getting, configuring, unpacking, building and installing Opticks and
its externals requires unix tools including:

* bash shell
* mercurial hg 
* git 
* curl
* tar
* zip
* cmake 2.8.9+

CMake
-------

A rather recent cmake version is required. Check your version with::

    simon:~ blyth$ cmake --version
    cmake version 3.4.1

Updating build tools is best done via your system package manager.  
For example on OSX with macports update cmake with::

   port info cmake           # check the version the package manager proposes
   sudo port install cmake   # do the install

If you or your system administrator are unable to update a tool via the system
package manager then a local install of the tool must be done and your 
login shell PATH modified to use the updated tool. The Opticks repository 
includes bash functions for local installs of cmake with 
precursor function *cmake-*.


Full Building Example
------------------------

Assuming appropriate build tools and Boost, CUDA (includes Thrust) and OptiX 
are already installed the getting, building and installation of the other externals 
takes less then 10 minutes and the Opticks build takes less than 5 minutes.::

    simon:env blyth$ opticks-fullclean | sh 
    simon:env blyth$ opticks- ; opticks-full
    === opticks-- : START Tue Apr 26 15:33:27 CST 2016
    === opticks-externals-install : START Tue Apr 26 15:33:27 CST 2016
    ...
    === opticks-externals-install : DONE Tue Apr 26 15:41:22 CST 2016
    ...
    === opticks-- : DONE Tue Apr 26 15:45:59 CST 2016


Externals 
-----------

Geometry/OpenGL related 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the bash function *opticks-externals-install*::

   opticks-externals-install

This gets the repositories or tarballs and perform the builds and installation.
Tools like hg, git, curl, tar, zip are assumed to be in your PATH.

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
graphics/assimp        assimp-          Assimp          github.com/simoncblyth/assimp fork of unspecified github version that handles G4DAE extras 
graphics/openmesh      openmesh-        OpenMesh        www.openmesh.org OpenMesh 4.1 tarball 
graphics/glm           glm-             GLM             sourceforge tarball 0.9.6.3, header only
graphics/glew          glew-            GLEW            sourceforge tarball 1.12.0, OpenGL extensions loading library   
graphics/glfw          glfw-            GLFW            sourceforge tarball 3.1.1, library for creating windows with OpenGL and receiving input   
graphics/gleq          gleq-            GLEQ            github.com/simoncblyth/gleq : GLFW author event handling example
graphics/gui/imgui     imgui-                           github.com/simoncblyth/imgui
=====================  ===============  =============   ==============================================================================


Boost Infrastructure Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pre-requisite Boost components listed in the table need to be installed.
These are widely available via package managers. Use the standard one for 
your system: 

* yum on Redhat
* macports on Mac
* nsys2 on Windows. 

The FindBoost.cmake provided with cmake is used to locate the installation.

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
boost                  boost-           Boost           components: system thread program_options log log_setup filesystem regex 
=====================  ===============  =============   ==============================================================================

Updating Boost 
~~~~~~~~~~~~~~~~

If your version of Boost is not recent enough the cmake configuring 
step will yield errors like the below.::

      CMake Error at /home/blyth/local/env/tools/cmake/cmake-3.5.2-Linux-x86_64/share/cmake-3.5/Modules/FindBoost.cmake:1657 (message):
      Unable to find the requested Boost libraries.

      Boost version: 1.41.0

      Boost include path: /usr/include

      Could not find the following Boost libraries:

              boost_log
              boost_log_setup

      Some (but not all) of the required Boost libraries were found.  You may
      need to install these additional Boost libraries.  Alternatively, set
      BOOST_LIBRARYDIR to the directory containing Boost libraries or BOOST_ROOT
      to the location of Boost.
      Call Stack (most recent call first):
      cmake/Modules/FindOpticksBoost.cmake:16 (find_package)
      boost/bpo/bcfg/CMakeLists.txt:8 (find_package)


If possible use your system package manager to update Boost. If that is 
not possible then do a local Boost install.  Opticks includes bash functions
starting *boost-* that can get and install Boost locally.


::

    opticks-
    boost-
    opticks-configure -DBOOST_ROOT=$(boost-prefix)



CUDA related
~~~~~~~~~~~~~

OptiX requires your system to have a fairly recent NVIDIA GPU of CUDA compute capability 3.0 at least.
However without such a GPU the OpenGL visualization should still work, using saved propagations. 

To download OptiX you need to join the NVIDIA Developer Program.  
Use the links in the table to register, it is free but may take a few days to be approved.
Follow the NVIDIA instructions to download and install CUDA and OptiX. 
Thrust is installed together with CUDA. 

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
cuda                   cuda-            CUDA            https://developer.nvidia.com/cuda-downloads
optix                  optix-           OptiX           https://developer.nvidia.com/optix
numerics/thrust        thrust-          Thrust          included with CUDA
=====================  ===============  =============   ==============================================================================


Configuring and Building Opticks
---------------------------------

CMake is used to configure Opticks and generate Makefiles. 
For a visualization only build with system Boost 
the defaults should work OK and there is no need to explicitly configure. 
If a local Boost was required then::

    opticks-configure -DBOOST_ROOT=$(boost-prefix) 
    
For a full build with CUDA and OptiX configure with::

    opticks-configure -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-7.0 \
                      -DOptiX_INSTALL_DIR=/Developer/OptiX \
                      -DBOOST_ROOT=$(boost-prefix) 
    

These configuration values are cached in the CMakeCache.txt file
in the build directory. These values are not overridden by rebuilding 
with the *opticks--* bash function. 
A subsequent *opticks-configure* however will wipe the build directory 
allowing new values to be set.


To build::
    opticks--


Configuration Machinery
------------------------

If the above configuration suceeded for you then 
you do not need to understand this machinery.

The below commands from the *opticks-cmake* bash function 
change directory to the build folder and invokes cmake 
to generate a configuration cache file and multiple Makefiles.::

   opticks-bcd
   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(optix-prefix) \
       $* \
       $(opticks-sdir)

CMake is controlled via CMakeLists.txt files. 
The top level one includes the below lines that 
locate the CUDA and OptiX:: 

    set(OPTICKS_CUDA_VERSION 5.5)
    set(OPTICKS_OPTIX_VERSION 3.5)
    ...
    find_package(CUDA ${OPTICKS_CUDA_VERSION})
    find_package(OptiX ${OPTICKS_OPTIX_VERSION})


Building Opticks 
---------------------

To build Opticks run::

   opticks-
   opticks-full   

After the first full build, faster update builds can be done with::

   opticks--

Full Opticks functionality with GPU simulation of optical photons requires all
the above externals to be installed, however if your GPU is not able to run OptiX or 
the CUDA related externals have not been installed it is still possible to make a 
partial build of Opticks using cmake switch WITH_OPTIX=OFF. 
The partial mode provides OpenGL visualizations of geometry and  
photon propagations loaded from file.


=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        required find package 
=====================  ===============  =============   ==============================================================================
boost/bpo/bcfg         bcfg-            BCfg            Boost
boost/bregex           bregex-          BRegex          Boost
numerics/npy           npy-             NPY             Boost GLM BRegex 
optickscore            optickscore-     OpticksCore     Boost GLM BRegex BCfg NPY 
optix/ggeo             ggeo-            GGeo            Boost GLM BRegex BCfg NPY OpticksCore
graphics/assimprap     assimprap-       AssimpRap       Boost GLM Assimp GGeo NPY OpticksCore
graphics/openmeshrap   openmeshrap-     OpenMeshRap     Boost GLM NPY GGeo OpticksCore OpenMesh 
opticksgeo             opticksgeo-      OpticksGeo      Boost GLM BRegex BCfg NPY OpticksCore Assimp AssimpRap OpenMesh OpenMeshRap
graphics/oglrap        oglrap-          OGLRap          GLEW GLFW GLM Boost BCfg Opticks GGeo NPY BRegex ImGui        
cuda/cudarap           cudarap-         CUDARap         CUDA (ssl)
numerics/thrustrap     thrustrap-       ThrustRap       CUDA Boost GLM NPY CUDARap 
graphics/optixrap      optixrap-        OptiXRap        OptiX CUDA Boost GLM NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap 
opticksop              opticksop-       OpticksOp       OptiX CUDA Boost GLM BCfg Opticks GGeo NPY OptiXRap CUDARap ThrustRap      
opticksgl              opticksgl-       OpticksGL       OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap OptiXRap OpticksOp
graphics/ggeoview      ggv-             GGeoView        OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY BCfg OpticksCore 
                                                        Assimp AssimpRap OpenMesh OpenMeshRap GGeo ImGui BRegex OptiXRap CUDARap ThrustRap OpticksOp OpticksGL OpticksGeo
optix/cfg4             cfg4-            CfG4            Boost GLM BRegex BCfg NPY GGeo OpticksCore Geant4 EnvXercesC G4DAE 
=====================  ===============  =============   ==============================================================================


bcfg
    commandline parsing 
bregex
    regular expression matching
npy
    array handling 
optickscore
    definitions, loosely the model of the app 
ggeo
    geometry representation 
assimprap
    G4DAE parsing into GGeo repr 
openmeshrap
    geometry fixing
opticksgeo
    bring together ggeo, assimprap and openmeshrap to load and fix geometry
oglrap
    OpenGL rendering, including GLSL shader sources
cudarap
    loading curand persisted state
thrustrap
    fast GPU photon indexing using interop techniques 
optixrap
    conversion of GGeo geometry into OptiX GPU geometry, OptiX programs for propagation 
opticksop
    high level OptiX control 
opticksgl 
    combination of oglrap- OpenGL and OptiX raytracing 
    TODO: change name ?
ggeoview
    putting together all the above
cfg4
    contained geant4 

     







 



EOU
}


opticks-env(){      
   elocal-
   g4- 
}

opticks-home(){   echo $(env-home) ; }
opticks-dir(){    echo $(local-base)/opticks ; }
opticks-prefix(){ echo $(local-base)/opticks ; }
opticks-sdir(){   echo $(opticks-home) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/bin ; }
opticks-xdir(){ echo $(opticks-prefix)/externals ; }

opticks-optix-install-dir(){ echo /Developer/OptiX ; }

opticks-cd(){   cd $(opticks-dir); }
opticks-scd(){  cd $(opticks-sdir); }
opticks-cd(){   cd $(opticks-sdir); }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }

opticks-wipe(){
   local bdir=$(opticks-bdir)
   rm -rf $bdir
}



opticks-cmake-generator()
{
    case $(uname -s) in
       MINGW64_NT*)  echo MSYS Makefiles ;;
                 *)  echo Unix Makefiles ;;
    esac                          
}


opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)
   mkdir -p $bdir

   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already  && return  

   opticks-bcd

   cmake \
        -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       $* \
       $(opticks-sdir)

   cd $iwd
}

opticks-configure(){
   opticks-wipe
   opticks-cmake $*
}


opticks-export()
{
  local bindir=$(opticks-prefix)/bin
  if [ "${PATH/$bindir}" == "${PATH}" ]; then
      export PATH=$bindir:$PATH
  fi   

  case $(uname -s) in
     MINGW*) opticks-export-mingw ;;
  esac
}

opticks-export-mingw()
{
  local dlldir=$(opticks-prefix)/lib
  if [ "${PATH/$dlldir}" == "${PATH}" ]; then
      export PATH=$dlldir:$PATH
  fi   

  # see bregex-/fsutil
  export OPTICKS_PATH_PREFIX="C:\\msys64" 
}


opticks-configure-local-boost(){
    local msg="=== $FUNCNAME :"
    boost-
    local prefix=$(boost-prefix)

    [ ! -d "$prefix" ] && type $FUNCNAME && return  
    echo $msg prefix $prefix
    opticks-configure -DBOOST_ROOT=$prefix
}


opticks-make(){
   local iwd=$PWD

   opticks-bcd
   make $*

   cd $iwd
}


opticks--(){
  ( opticks-bcd ; make ${1:-install} )
}

opticks-ctest(){
  ( opticks-bcd ; ctest $* ; )
}



opticks-full()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)

    if [ ! -d "$(opticks-prefix)/externals" ]; then
         opticks-externals-install
    fi 

    opticks-cmake $*
    opticks-make install

    echo $msg DONE $(date)
}






opticks-externals-install(){
   local msg="=== $FUNCNAME :"

   local exts="glm glfw glew gleq imgui assimp openmesh"

   echo $msg START $(date)

   local ext
   for ext in $exts 
   do
        echo $msg $ext
        $ext-
        $ext--
   done

   echo $msg DONE $(date)
}





opticks-bin(){ echo $(opticks-idir)/bin/GGeoView ; }

opticks-run()
{

    export-
    export-export   ## needed to setup DAE_NAME_DYB the envvar name pointed at by the default opticks_GEOKEY 

    local bin=$(opticks-bin)
    $bin $*         ## bare running with no bash script, for checking defaults 

}





opticks-libtyp()
{
   case $(uname -s) in
     Darwin) echo dylib ;;
     Linux)  echo so ;;
         *)  echo dll ;;
   esac
}




########## below are for development  ########################





opticks-dirs(){  cat << EOL
boost/bpo/bcfg
boost/bregex
numerics/npy
optickscore
optix/ggeo
graphics/assimprap
graphics/openmeshrap
graphics/oglrap
cuda/cudarap
numerics/thrustrap
graphics/optixrap
opticksop
opticksgl
graphics/ggeoview
optix/cfg4
EOL
}
opticks-internals(){  cat << EOI
BCfg
BRegex
NPY
OpticksCore
GGeo
AssimpRap
OpenMeshRap
OGLRap
CUDARap
ThrustRap
OptiXRap
OpticksOp
OpticksGL
OptiXThrust
NumpyServer
EOI
}
opticks-xternals(){  cat << EOX
OpticksBoost
Assimp
OpenMesh
GLM
GLEW
GLEQ
GLFW
ImGui

EnvXercesC
G4DAE
ZMQ
AsioZMQ
EOX
}
opticks-other(){  cat << EOO
OpenVR
CNPY
NuWaCLHEP
NuWaGeant4
cJSON
RapSqlite
SQLite3
ChromaPhotonList
G4DAEChroma
NuWaDataModel
ChromaGeant4CLHEP
CLHEP
ROOT
ZMQRoot
EOO
}


opticks-find-cmake-(){ 
  local f
  local base=$(opticks-home)/CMake/Modules
  local name
  opticks-${1} | while read f 
  do
     name=$base/Find${f}.cmake
     [ -f "$name" ] && echo $name
  done 
}

opticks-i(){ vi $(opticks-find-cmake- internals) ; }
opticks-x(){ vi $(opticks-find-cmake- xternals) ; }
opticks-o(){ vi $(opticks-find-cmake- other) ; }



opticks-distclean(){ opticks-rmdirs- bin build gl include lib ptx  ; }
opticks-fullclean(){ opticks-rmdirs- bin build gl include lib ptx externals  ; }

opticks-rmdirs-(){
   local base=$(opticks-dir)
   local msg="# $FUNCNAME : "
   echo $msg pipe to sh to do the deletion
   local name
   for name in $*
   do 
      local dir=$base/$name
      [ -d "$dir" ] && echo rm -rf $dir ;
   done
}



opticks-edit(){  cd $ENV_HOME ; vi opticks.bash $(opticks-bash-list) CMakeLists.txt $(opticks-txt-list) ; } 
opticks-txt(){   cd $ENV_HOME ; vi CMakeLists.txt $(opticks-txt-list) ; }
opticks-bash(){  cd $ENV_HOME ; vi opticks.bash $(opticks-bash-list) ; }
opticks-tests(){ cd $ENV_HOME ; vi $(opticks-tests-list) ; } 

opticks-txt-list(){
  local dir
  opticks-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
  done
}
opticks-tests-list(){
  local dir
  local name
  opticks-dirs | while read dir 
  do
      name=$dir/tests/CMakeLists.txt
      [ -f "$name" ] && echo $name
  done

}
opticks-bash-list(){
  local dir
  opticks-dirs | while read dir 
  do
      local rel=$dir/$(basename $dir).bash
      if [ -f "$rel" ]; 
      then
          echo $rel
      else
          echo MISSING $rel
      fi
  done
}


# not needed as RPATH is working 
#opticks-ldpath(){ 
#   boost-
#   assimp-
#   openmesh-
#   glew-
#   glfw-
#   imgui-
#   echo "$(boost-prefix)/lib;$(opticks-prefix)/lib;$(assimp-prefix)/lib;$(openmesh-prefix)/lib;$(glew-prefix)/lib64;$(glfw-prefix)/lib;$(imgui-prefix)/lib"
#}


opticks-check(){ 
   local msg="=== $FUNCNAME :"
   local dae=$HOME/g4_00.dae
   [ ! -f "$dae" ] && echo $msg missing geometry file $dae && return 
   $(opticks-prefix)/bin/GGeoView --size 1024,768,1 $dae
}


