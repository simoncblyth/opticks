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

See Also
----------

cmake-
    background on cmake

cmakex-
    documenting the development of the opticks- cmake machinery 


Fullbuild Testing
------------------

Only needed whilst making sweeping changes::

    simon:~ blyth$ opticks-distclean         # check what will be deleted
    simon:~ blyth$ opticks-distclean | sh    # delete 

    simon:~ blyth$ opticks-fullclean         # check what will be deleted
    simon:~ blyth$ opticks-fullclean | sh    # delete 

    simon:~ blyth$ opticks- ; opticks--


G4PB build
-----------

* /usr/local/opticks : permission denied

G5 build
----------

* glfw:CMake 2.8.9 or higher is required.  You are running version 2.6.4


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
    simon:env blyth$ opticks- ; opticks--
    === opticks-- : START Tue Apr 26 15:33:27 CST 2016
    === opticks-externals-install : START Tue Apr 26 15:33:27 CST 2016
    ...
    === opticks-externals-install : DONE Tue Apr 26 15:41:22 CST 2016
    ...
    === opticks-- : DONE Tue Apr 26 15:45:59 CST 2016


Externals 
-----------

Infrastructure 
~~~~~~~~~~~~~~~~

The pre-requisite Boost components listed in the table need to be installed.
These are widely available via package managers. Use the standard one for 
your system: yum on Redhat, macports on Mac, nsys2 on Windows. 
The FindBoost.cmake provided with cmake is used to locate the installation.

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
boost                  boost-           Boost           components: system thread program_options log log_setup filesystem regex 
=====================  ===============  =============   ==============================================================================



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

CUDA related
~~~~~~~~~~~~~

OptiX requires your system to have a fairly recent NVIDIA GPU of CUDA compute capability 3.0 at least.
However without such a GPU the OpenGL visualization should still work, using saved propagations. 

To download OptiX you need to join the NVIDIA Developer Program.  
Use the links in the table to register, it is free but may take a few days to be approved.
Follow the NVIDIA instructions to download and install CUDA and OptiX. 
Thrust is installed together with CUDA. 
The cmake provided FindCUDA.cmake is used to locate the installation.

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
cuda                   cuda-            CUDA            https://developer.nvidia.com/cuda-downloads
optix                  optix-           OptiX           https://developer.nvidia.com/optix
numerics/thrust        thrust-          Thrust          included with CUDA
=====================  ===============  =============   ==============================================================================


Building Opticks 
---------------------

To build Opticks run::

   opticks-
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
graphics/assimprap     assimprap-       AssimpRap       Boost Assimp GGeo GLM NPY OpticksCore
graphics/openmeshrap   openmeshrap-     OpenMeshRap     Boost GLM NPY GGeo OpticksCore OpenMesh 
graphics/oglrap        oglrap-          OGLRap          GLEW GLFW GLM Boost BCfg Opticks GGeo NPY BRegex ImGui        
cuda/cudarap           cudarap-         CUDARap         CUDA (ssl)
numerics/thrustrap     thrustrap-       ThrustRap       CUDA Boost GLM NPY CUDARap 
graphics/optixrap      optixrap-        OptiXRap        OptiX CUDA Boost GLM NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap 
opticksop              opticksop-       OpticksOp       OptiX CUDA Boost GLM BCfg Opticks GGeo NPY OptiXRap CUDARap ThrustRap      
opticksgl              opticksgl-       OpticksGL       OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap OptiXRap OpticksOp
graphics/ggeoview      ggv-             GGeoView        OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY BCfg OpticksCore 
                                                        Assimp AssimpRap OpenMesh OpenMeshRap GGeo ImGui BRegex OptiXRap CUDARap ThrustRap OpticksOp OpticksGL 
optix/cfg4             cfg4-            CfG4            Boost BRegex GLM NPY BCfg GGeo OpticksCore Geant4 EnvXercesC G4DAE 
=====================  ===============  =============   ==============================================================================



EOU
}


opticks-env(){      
   elocal- 
}

opticks-home(){   echo $(env-home) ; }
opticks-dir(){    echo $(local-base)/opticks ; }
opticks-prefix(){ echo $(local-base)/opticks ; }
opticks-sdir(){   echo $(opticks-home) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }

opticks-optix-install-dir(){ echo /Developer/OptiX ; }

opticks-cd(){   cd $(opticks-dir); }
opticks-scd(){  cd $(opticks-sdir); }
opticks-cd(){   cd $(opticks-sdir); }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }


opticks-wipe(){
   local bdir=$(opticks-bdir)
   rm -rf $bdir
}

opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)
   mkdir -p $bdir

   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already  && return  

   opticks-bcd

   cmake \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-idir) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       $(opticks-sdir)

   cd $iwd
}

opticks-configure(){
   opticks-wipe
   opticks-cmake
}

opticks-make(){
   local iwd=$PWD

   opticks-bcd
   make $*

   cd $iwd
}


opticks--()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)

    if [ ! -d "$(opticks-prefix)/externals" ]; then
         opticks-externals-install
    fi 

    opticks-cmake
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
Boost
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


opticks-tfind-(){ 
  local f
  local base=$(opticks-home)/CMake/Modules
  local name
  opticks-${1} | while read f 
  do
     name=$base/Find${f}.cmake
     [ -f "$name" ] && echo $name
  done 
}

opticks-ifind(){ vi $(opticks-tfind- internals) ; }
opticks-xfind(){ vi $(opticks-tfind- xternals) ; }
opticks-ofind(){ vi $(opticks-tfind- other) ; }



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



