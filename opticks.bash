opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-usage(){   cat << \EOU

Opticks
=========

Get Opticks 
-----------

Clone the repository from bitbucket::

   cd
   hg clone http://bitbucket.org/simoncblyth/env

In future Opticks is intended to move, then you will need to::

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
graphics/glm           glm-             GLM             sourceforge tarball 0.9.6.3, header only
graphics/assimp        assimp-          Assimp          github.com/simoncblyth/assimp fork of unspecified github version that handles G4DAE extras, cmake configured
graphics/openmesh      openmesh-        OpenMesh        www.openmesh.org OpenMesh 4.1 tarball, cmake configured, provides VS2015 binaries http://www.openmesh.org/download/
graphics/glew          glew-            GLEW            sourceforge tarball 1.12.0, OpenGL extensions loading library, cmake build didnt work, includes vc12 sln for windows
graphics/glfw          glfw-            GLFW            sourceforge tarball 3.1.1, library for creating windows with OpenGL and receiving input, cmake generation    
graphics/gleq          gleq-            GLEQ            github.com/simoncblyth/gleq : GLFW author event handling example, header only
graphics/gui/imgui     imgui-                           github.com/simoncblyth/imgui expected to drop source into using project, simple CMakeLists.txt added by me
=====================  ===============  =============   ==============================================================================


Dependencies of externals:

============  ====================  ==============================
pkg           depends on            notes  
============  ====================  ==============================
glm                                 headers only  
gleq                                headers only  
glew          system opengl
glfw
imgui         glfw glew 
============  ====================  ==============================



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
sysrap                 sysrap-          SysRap          PLog
boostrap               brap-            BoostRap        OpticksBoost
numerics/npy           npy-             NPY             Boost GLM BoostRap
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


sysrap
    logging, string handling, envvar handling 
boostrap
    filesystem utils, regular expression matching, commandline parsing 
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
   # dont pollute : otherwise will get infinite loops : as opticks is used in many other -env
}
opticks-home(){   echo $(env-home) ; }
opticks-dir(){    echo $(local-base)/opticks ; }
opticks-prefix(){ echo $(local-base)/opticks ; }
opticks-sdir(){   echo $(opticks-home) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/lib ; }   # use lib for executables for simplicity on windows
opticks-xdir(){ echo $(opticks-prefix)/externals ; }

opticks-optix-install-dir(){ echo /Developer/OptiX ; }

opticks-cd(){   cd $(opticks-dir) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }


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




opticks-cmake-generator()
{
    if [ "$NODE_TAG" == "M" ]; then
       echo MSYS Makefiles 
    else  
       case $(uname -s) in
         MINGW64_NT*)  echo Visual Studio 14 2015 ;;
                   *)  echo Unix Makefiles ;;
       esac                          
    fi

}

opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD
   local bdir=$(opticks-bdir)
   mkdir -p $bdir
   [ -f "$bdir/CMakeCache.txt" ] && echo $msg configured already use opticks-configure to reconfigure  && return  
   opticks-bcd
   g4- 
   xercesc-

   cmake \
        -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
       $* \
       $(opticks-sdir)

   cd $iwd
}

opticks-cmake-modify(){
  local msg="=== $FUNCNAME : "
  local bdir=$(opticks-bdir)
  local bcache=$bdir/CMakeCache.txt
  [ ! -f "$bcache" ] && echo $msg requires a preexisting $bcache from prior opticks-cmake run && return 
  opticks-bcd
  g4-
  xercesc- 
  cmake \
       -DGeant4_DIR=$(g4-cmake-dir) \
       -DXERCESC_LIBRARY=$(xercesc-library) \
       -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir) \
          . 
}


opticks-wipe(){
   local bdir=$(opticks-bdir)
   rm -rf $bdir
}

opticks-configure(){
   opticks-wipe
   case $(opticks-cmake-generator) in
       "Visual Studio 14 2015") opticks-configure-local-boost $* ;;
                             *) opticks-configure-system-boost $* ;;
   esac
}

opticks-configure-system-boost(){
   opticks-cmake $* 
}

opticks-configure-local-boost(){
    type $FUNCNAME

    local msg="=== $FUNCNAME :"
    boost-

    local prefix=$(boost-prefix)
    [ ! -d "$prefix" ] && type $FUNCNAME && return  
    echo $msg prefix $prefix

    opticks-cmake \
              -DBOOST_ROOT=$prefix \
              -DBoost_USE_STATIC_LIBS=1 \
              -DBoost_USE_DEBUG_RUNTIME=0 \
              -DBoost_NO_SYSTEM_PATHS=1 \
              -DBoost_DEBUG=0 

    # vi $(cmake-find-package Boost)

}



opticks-name(){ echo Opticks ; }
opticks-sln(){ echo $(opticks-bdir)/$(opticks-name).sln ; }
opticks-slnw(){  vs- ; echo $(vs-wp $(opticks-sln)) ; }
opticks-vs(){ 
   vs-
   local sln=$1
   [ -z "$sln" ] && sln=$(opticks-sln) 
   local slnw=$(vs-wp $sln)

    cat << EOC
# sln  $sln
# slnw $slnw
# copy/paste into powershell v2 OR just use opticks-vs Powershell function
vs-export 
devenv /useenv $slnw
EOC

}
   

#opticks-config(){ echo Debug ; }
opticks-config(){ echo RelWithDebInfo ; }
opticks--(){     

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" -o "$bdir" == "." ] && bdir=$(opticks-bdir) 
   [ ! -d "$bdir" ] && echo $msg bdir $bdir does not exist && return 

   cd $bdir

   cmake --build . --config $(opticks-config) --target ${1:-install}

   cd $iwd
}


opticks-ctest()
{
   # 
   # Basic environment (PATH and envvars to find data) 
   # should happen at profile level (or at least be invoked from there) 
   # not here (and there) for clarity of a single location 
   # where smth is done.
   #
   # Powershell presents a challenge to this principal,
   # TODO:find a cross platform way of doing envvar setup 
   #
   #

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" ] && bdir=$(opticks-bdir) 

   cd $bdir

   ctest $*


   cd $iwd
   echo $msg use -V to show output 
}



opticks-ctest-deprecated()
{ 

   local msg="$FUNCNAME : "
   local iwd=$PWD

   local bdir=$1
   shift
   [ -z "$bdir" ] && bdir=$(opticks-bdir) 

   cd $bdir

   #export-
   #export-export 

   opticksdata-
   opticksdata-export


   if [ "$USERPROFILE" == "" ]; then 
      ctest $*   
   else
      # windows needs PATH to find libs
      PATH=$(opticks-prefix)/lib:$PATH ctest $*   
   fi

   cd $iwd

   echo $msg use -V to show output 
}


opticks---(){ 

  sysrap-
  sysrap--

  brap-
  brap--

  npy-
  npy--

  okc-
  okc--

  ggeo-
  ggeo--

  assimprap-
  assimprap--

  openmeshrap-
  openmeshrap--

  opticksgeo-
  opticksgeo--

  oglrap-
  oglrap--

  ############ CUDA NEEDED 
  cudarap-
  cudarap--

  thrustrap- 
  thrustrap--

  optixrap-
  optixrap--

  opticksop-
  opticksop--

  opticksgl-
  opticksgl--
  ####################### 
 
  ggeoview-
  ggeoview--

  cfg4-
  cfg4--


} 


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

opticks-full()
{
    local msg="=== $FUNCNAME :"
    echo $msg START $(date)

    if [ ! -d "$(opticks-prefix)/externals" ]; then
         opticks-externals-install
    fi 

    opticks-configure

    opticks--

    echo $msg DONE $(date)
}

opticks-cleanbuild()
{
   opticks-distclean 
   opticks-distclean | sh 
   opticks-full 
}


########## runtime setup ########################

opticks-check(){ 
   # last arg dae running is not the usual approach 
   local msg="=== $FUNCNAME :"
   local dae=$HOME/g4_00.dae
   [ ! -f "$dae" ] && echo $msg missing geometry file $dae && return 
   $(opticks-prefix)/bin/GGeoView --size 1024,768,1 $dae
}

opticks-path(){ echo $PATH | tr ":" "\n" ; }
opticks-path-add(){
  local dir=$1 
  [ "${PATH/$dir}" == "${PATH}" ] && export PATH=$dir:$PATH
}

opticks-export()
{
   opticks-export-common

   case $(uname -s) in
      MINGW*) opticks-export-mingw ;;
   esac
}
opticks-export-common()
{
   opticks-path-add $(opticks-prefix)/bin

   opticksdata-
   opticksdata-export

}
opticks-export-mingw()
{
  local dirs="lib externals/bin externals/lib"
  local dir
  for dir in $dirs 
  do
      opticks-path-add $(opticks-prefix)/$dir
  done 

  # see brap-/fsutil
  export OPTICKS_PATH_PREFIX="C:\\msys64" 
}


########## below are for development  ########################


opticks-dirs(){  cat << EOL
sysrap
boostrap
opticksnpy
optickscore
ggeo
assimprap
openmeshrap
opticksgeo
oglrap
cudarap
thrustrap
optixrap
opticksop
opticksgl
graphics/ggeoview
optix/cfg4
EOL
}

opticks-xnames(){ cat << EOX
boost
glm
plog
gleq
glfw
glew
imgui
assimp
openmesh
cuda
thrust
optix
xercesc
g4
EOX
}

opticks-internals(){  cat << EOI
SysRap
BoostRap
NPY
OpticksCore
GGeo
AssimpRap
OpenMeshRap
OpticksGeo
OGLRap
CUDARap
ThrustRap
OptiXRap
OpticksOp
OpticksGL
GGeoView
CfG4
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
opticks-grep()
{
   local iwd=$PWD
   local msg="=== $FUNCNAME : "
   opticks-
   local dir 
   local base=$(opticks-home)
   opticks-dirs | while read dir 
   do
      local subdirs="${base}/${dir} ${base}/${dir}/tests"
      local sub
      for sub in $subdirs 
      do
         if [ -d "$sub" ]; then 
            cd $sub
            #echo $msg $sub
            grep $* $PWD/*.*
         fi
      done 
   done
   cd $iwd
}


opticks-api-export()
{
  opticks-cd 
  local dir
  opticks-dirs | while read dir 
  do
      local name=$(ls -1 $dir/*_API_EXPORT.hh 2>/dev/null) 
      [ ! -z "$name" ] && echo $name
  done
}

opticks-api-export-vi(){ vi $(opticks-api-export) ; }




opticks-grep-vi(){ vi $(opticks-grep -l ${1:-BLog}) ; }



opticks-genproj()
{
    # this is typically called from projs like ggeo- 

    local msg=" === $FUNCNAME :"
    local proj=${1}
    local tag=${2}

    [ -z "$proj" -o -z "$tag" ] && echo $msg need both proj $proj and tag $tag  && return 


    importlib-  
    importlib-exports ${proj} ${tag}_API

    plog-
    plog-genlog

    echo $msg merge the below sources into CMakeLists.txt
    opticks-genproj-sources- $tag

}



opticks-genlog()
{
    opticks-scd 
    local dir
    plog-
    opticks-dirs | while read dir 
    do
        opticks-scd $dir

        local name=$(ls -1 *_API_EXPORT.hh 2>/dev/null) 
        [ -z "$name" ] && echo MISSING API_EXPORT in $PWD && return 
        [ ! -z "$name" ] && echo $name

        echo $PWD
        plog-genlog FORCE
    done
}


opticks-genproj-sources-(){ 


   local tag=${1:-OKCORE}
   cat << EOS

set(SOURCES
     
    ${tag}_LOG.cc

)
set(HEADERS

    ${tag}_LOG.hh
    ${tag}_API_EXPORT.hh
    ${tag}_HEAD.hh
    ${tag}_TAIL.hh

)
EOS
}


opticks-testname(){ echo ${cls}Test.cc ; }
opticks-gentest()
{
   local msg=" === $FUNCNAME :"
   local cls=${1:-GMaterial}
   local tag=${2:-GGEO} 

   [ -z "$cls" -o -z "$tag" ] && echo $msg a classname $cls and project tag $tag must be provided && return 
   local name=$(opticks-testname $cls)
   [ -f "$name" ] && echo $msg a file named $name exists already in $PWD && return
   echo $msg cls $cls generating test named $name in $PWD
   opticks-gentest- $cls $tag > $name
   #cat $name

   vi $name

}
opticks-gentest-(){

   local cls=${1:-GMaterial}
   local tag=${2:-GGEO}

   cat << EOT

#include <cassert>
#include "${cls}.hh"

#include "PLOG.hh"
#include "${tag}_LOG.hh"

int main(int argc, char** argv)
{
    PLOG_(argc, argv);
    ${tag}_LOG_ ;




    return 0 ;
}

EOT

}

opticks-xcollect-notes(){ cat << EON

*opticks-xcollect*
     copies the .bash of externals into externals folder 
     and does inplace edits to correct paths for new home.
     Also writes an externals.bash containing the precursor bash 
     functions.

EON
}
opticks-xcollect()
{
   local ehome=$(env-home)
   local xhome=$ehome
   local iwd=$PWD 

   cd $ehome

   local xbash=$xhome/externals/externals.bash
   [ ! -d "$xhome/externals" ] && mkdir "$xhome/externals"

   echo "# $FUNCNAME " > $xbash
   
   local x
   local esrc
   local src
   local dst
   local nam

   opticks-xnames | while read x 
   do
      $x-;
      esrc=$($x-source)
      src=${esrc/$ehome\/}
      nam=$(basename $src)
      dst=externals/$nam

      printf "# %-15s %15s %35s \n" $x $nam $src

      cp $src $xhome/$dst
      perl -pi -e "s,$src,$dst," $xhome/$dst 
      perl -pi -e "s,env-home,opticks-home," $xhome/$dst 

      printf "%-20s %-50s %s\n" "$x-(){" ". \$(opticks-home)/externals/$nam" "&& $x-env \$* ; }"   >> $xbash

   done 
   cd $iwd
}

