opticks-(){         source $(opticks-source) && opticks-env $* ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(opticks-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-usage(){   cat << \EOU

Opticks Install Instructions
==================================

Using a Shared Opticks Installation
-------------------------------------

If someone has installed Opticks for you already 
you just need to set the PATH variable in your .bash_profile 
to easily find the Opticks executables and scripts. 

.. code-block:: sh

    # .bash_profile

    # Get the aliases and functions
    if [ -f ~/.bashrc ]; then
        . ~/.bashrc
    fi

    # User specific environment and startup programs

    PATH=$PATH:$HOME/.local/bin:$HOME/bin
    ini(){ . ~/.bash_profile ; }

    ok-local(){    echo /home/simonblyth/local ; }
    ok-opticks(){  echo /home/simonblyth/opticks ; }
    ok-ctest(){    ( cd $(ok-local)/opticks/build ; ctest3 $* ; ) }

    export PATH=$(ok-opticks)/bin:$(ok-local)/opticks/lib:$PATH


You can test the installation using the `ok-ctest` function defined in 
the .bash_profile. The output shoule look like the below. 
The permission denied error is not a problem.

.. code-block:: sh

    [blyth@optix ~]$ ok-ctest
    Test project /home/simonblyth/local/opticks/build
    CMake Error: Cannot open file for write: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log.tmp
    CMake Error: : System Error: Permission denied
    Problem opening file: /home/simonblyth/local/opticks/build/Testing/Temporary/LastTest.log
    Cannot create log file: LastTest.log
            Start   1: SysRapTest.SEnvTest
      1/155 Test   #1: SysRapTest.SEnvTest ........................   Passed    0.00 sec
            Start   2: SysRapTest.SSysTest
    ...
    ...
    154/155 Test #154: cfg4Test.G4StringTest ......................   Passed    0.06 sec
            Start 155: cfg4Test.G4BoxTest
    155/155 Test #155: cfg4Test.G4BoxTest .........................   Passed    0.05 sec

    100% tests passed, 0 tests failed out of 155

    Total Test time (real) =  48.30 sec



Get Opticks 
------------

Clone the repository from bitbucket::

   hg clone http://bitbucket.org/simoncblyth/opticks 

Connect the opticks bash functions to your shell by adding a line to your .bash_profile
and configure the location of the install with the LOCAL_BASE environment variable::

   opticks-(){ . $HOME/opticks/opticks.bash && opticks-env $* ; }
   export LOCAL_BASE=/usr/local   

The first line defines the bash function *opticks-* that is termed a precursor function 
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

    simon:env blyth$ opticks-fullclean | sh   ## deletes dirs beneath $LOCAL_BASE/opticks
    simon:env blyth$ opticks- ; opticks-full


Externals 
-----------

Use the bash function *opticks-externals-install*::

   opticks-externals-install

This gets the repositories or tarballs and perform the builds and installation.
Tools like hg, git, curl, tar, zip are assumed to be in your PATH.

===============  =============   ==============================================================================
precursor        pkg name        notes
===============  =============   ==============================================================================
glm-             GLM             OpenGL mathematics, 3D transforms 
assimp-          Assimp          Assimp 3D asset importer, my fork that handles G4DAE extras
openmesh-        OpenMesh        basis for mesh navigation and fixing
glew-            GLEW            OpenGL extensions loading library, cmake build didnt work, includes vc12 sln for windows
glfw-            GLFW            Interface between system and OpenGL, creating windows and receiving input
gleq-            GLEQ            Keyboard event handling header from GLFW author, header only
imgui-           ImGui           OpenGL immediate mode GUI, depends on glfw and glew
plog-            PLog            Header only logging, supporting multi dll logging on windows 
opticksdata-     -               Dayabay G4DAE and GDML geometry files for testing Opticks      
===============  =============   ==============================================================================


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
cuda                   cuda-            CUDA            https://developer.nvidia.com/cuda-downloads (includes Thrust)
optix                  optix-           OptiX           https://developer.nvidia.com/optix
=====================  ===============  =============   ==============================================================================


Configuring and Building Opticks
---------------------------------

CMake is used to configure Opticks and generate Makefiles or Visual Studio solution files on windows.
For a visualization only build with system Boost 
the defaults should work OK and there is no need to explicitly configure. 
If a local Boost was required then::

    opticks-configure -DBOOST_ROOT=$(boost-prefix) 
    
For a full build with CUDA and OptiX configure with::

    opticks-configure -DCUDA_TOOLKIT_ROOT_DIR=/Developer/NVIDIA/CUDA-7.0 \
                      -DOptiX_INSTALL_DIR=/Developer/OptiX \
                      -DCOMPUTE_CAPABILITY=52 \
                      -DBOOST_ROOT=$(boost-prefix) 


The argument `-DCOMPUTE_CAPABILITY=52` specifies to compile for compute capability 5.2 architectures 
corresponding to Maxwell 2nd generation GPUs. 
Lookup the appropriate capability for your GPU in the below short table.

====================  =========================  =================== 
Compute Capability    Architecture               GPU Examples
====================  =========================  ===================
2.1                   Fermi                      **NOT SUPPORTED BY OPTICKS**
3.0                   Kepler                     GeForce GT 750M
5.0                   Maxwell 1st generation     Quadro M2000M
5.2                   Maxwell 2nd generation     Quadro M5000
6.1                   Pascal                     GeForce GTX 1080
====================  =========================  ===================

For more complete tables see

* https://en.wikipedia.org/wiki/CUDA
* https://developer.nvidia.com/cuda-gpus.

Opticks requires a compute capability of at least 3.0, if you have no suitable GPU 
or would like to test without GPU acceleration use `-DCOMPUTE_CAPABILITY=0`.


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

Testing Installation
----------------------

The *opticks-ctest* functions runs ctests for all the opticks projects::

    simon:opticks blyth$ opticks-
    simon:opticks blyth$ opticks-ctest
    Test project /usr/local/opticks/build
          Start  1: SysRapTest.SEnvTest
     1/65 Test  #1: SysRapTest.SEnvTest ........................   Passed    0.00 sec
          Start  2: SysRapTest.SSysTest
     2/65 Test  #2: SysRapTest.SSysTest ........................   Passed    0.00 sec
          Start  3: SysRapTest.SDigestTest
     3/65 Test  #3: SysRapTest.SDigestTest .....................   Passed    0.00 sec
    .....
    ..... 
          Start 59: cfg4Test.CPropLibTest
    59/65 Test #59: cfg4Test.CPropLibTest ......................   Passed    0.05 sec
          Start 60: cfg4Test.CTestDetectorTest
    60/65 Test #60: cfg4Test.CTestDetectorTest .................   Passed    0.04 sec
          Start 61: cfg4Test.CGDMLDetectorTest
    61/65 Test #61: cfg4Test.CGDMLDetectorTest .................   Passed    0.45 sec
          Start 62: cfg4Test.CG4Test
    62/65 Test #62: cfg4Test.CG4Test ...........................   Passed    5.06 sec
          Start 63: cfg4Test.G4MaterialTest
    63/65 Test #63: cfg4Test.G4MaterialTest ....................   Passed    0.02 sec
          Start 64: cfg4Test.G4StringTest
    64/65 Test #64: cfg4Test.G4StringTest ......................   Passed    0.02 sec
          Start 65: cfg4Test.G4BoxTest
    65/65 Test #65: cfg4Test.G4BoxTest .........................   Passed    0.02 sec

    100% tests passed, 0 tests failed out of 65

    Total Test time (real) =  59.89 sec
    opticks-ctest : use -V to show output


Issues With Tests
-------------------

Some tests depend on the geometry cache being present. To create the geometry cache::

   op.sh -G 



Running Opticks Scripts and Executables
----------------------------------------

All Opticks executables including the tests are installed 
into $LOCAL_BASE/opticks/lib/ an example `.bash_profile` 
to is provided below:

.. code-block:: sh

    # .bash_profile

    if [ -f ~/.bashrc ]; then                 ## typical setup 
            . ~/.bashrc
    fi

    export NODE_TAG_OVERRIDE=X                ## env hookup only needed by Opticks developers
    export ENV_HOME=$HOME/env
    env-(){  [ -r $ENV_HOME/env.bash ] && . $ENV_HOME/env.bash && env-env $* ; }
    env-

    export LOCAL_BASE=$HOME/local             ## opticks hookup is needed by all Opticks users 
    export OPTICKS_HOME=$HOME/opticks

    opticks-(){  [ -r $HOME/opticks/opticks.bash ] && . $HOME/opticks/opticks.bash && opticks-env $* ; }
    opticks-                                  ## defines several bash functions beginning opticks- eg opticks-info

    o(){ cd $(opticks-home) ; hg st ; }
    op(){ op.sh $* ; }

    PATH=$OPTICKS_HOME/bin:$LOCAL_BASE/opticks/lib:$PATH  ## easy access to scripts and executables
    export PATH


EOU
}

opticks-env(){      
   # dont pollute : otherwise will get infinite loops : as opticks is used in many other -env
   . $(opticks-home)/externals/externals.bash   ## just precursors
}

olocal-()
{
   echo -n # transitional standin for olocal-
}

opticks-home(){   echo ${OPTICKS_HOME:-$HOME/opticks} ; }  ## input from profile 
opticks-dir(){    echo $(opticks-prefix) ; }

opticks-suffix(){
   case $(uname) in
      MING*) echo .exe ;;
          *) echo -n  ;;   
   esac
}

opticks-prefix(){ 
   # when LOCAL_BASE unset rely instead on finding an installed binary from PATH  
   if [ -z "$LOCAL_BASE" ]; then 
      echo $(dirname $(dirname $(which OpticksTest$(opticks-suffix))))
   else
      echo ${LOCAL_BASE}/opticks ;
   fi
}


opticks-sdir(){   echo $(opticks-home) ; }
opticks-idir(){   echo $(opticks-prefix) ; }
opticks-bdir(){   echo $(opticks-prefix)/build ; }
opticks-bindir(){ echo $(opticks-prefix)/lib ; }   # use lib for executables for simplicity on windows
opticks-xdir(){   echo $(opticks-prefix)/externals ; }

opticks-cd(){   cd $(opticks-dir) ; }
opticks-scd(){  cd $(opticks-sdir)/$1 ; }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }
opticks-xcd(){  cd $(opticks-xdir); }



opticks-optix-install-dir(){ 
    local t=$NODE_TAG
    case $t in 
       D) echo /Developer/OptiX ;;
     GTL) echo ${MYENVTOP}/OptiX ;;
    H5H2) echo ${MYENVTOP}/OptiX ;;
       X) echo /usr/local/optix-3.8.0/NVIDIA-OptiX-SDK-3.8.0-linux64 ;;
       *) echo /tmp ;;
    esac
}

opticks-compute-capability(){
    local t=$NODE_TAG
    case $t in 
       D) echo 30 ;;
     GTL) echo 30 ;;
    H5H2) echo 50 ;;
       X) echo 52 ;; 
       *) echo  0 ;;
    esac
}

opticks-externals(){ cat << EOL
glm
glfw
glew
gleq
imgui
assimp
openmesh
plog
opticksdata
EOL
}

opticks-optionals(){ cat << EOL
xercesc
g4
EOL
}

-opticks-installer(){
   local msg="=== $FUNCNAME :"
   echo $msg START $(date)
   local ext
   while read ext 
   do
        echo $msg $ext
        $ext-
        $ext--
   done
   echo $msg DONE $(date)
}

-opticks-url(){
   local ext
   while read ext 
   do
        $ext-
        printf "%30s :  %s \n" $ext $($ext-url) 
   done
}

-opticks-dist(){
   local ext
   local dist
   while read ext 
   do
        $ext-
        dist=$($ext-dist 2>/dev/null)
        printf "%30s :  %s \n" $ext $dist
   done
}

opticks-externals-install(){ opticks-externals | -opticks-installer ; }
opticks-externals-url(){     opticks-externals | -opticks-url ; }
opticks-externals-dist(){    opticks-externals | -opticks-dist ; }

opticks-optionals-install(){ opticks-optionals | -opticks-installer ; }
opticks-optionals-url(){     opticks-optionals | -opticks-url ; }
opticks-optionals-dist(){    opticks-optionals | -opticks-dist ; }

opticks-info(){
   echo externals-url
   opticks-externals-url
   echo externals-dist
   opticks-externals-dist
   echo optionals-url
   opticks-optionals-url
   echo optionals-dist
   opticks-optionals-dist
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


   echo $msg opticks-prefix:$(opticks-prefix)
   echo $msg opticks-optix-install-dir:$(opticks-optix-install-dir)
   echo $msg g4-cmake-dir:$(g4-cmake-dir)
   echo $msg xercesc-library:$(xercesc-library)
   echo $msg xercesc-include-dir:$(xercesc-include-dir)

   cmake \
        -G "$(opticks-cmake-generator)" \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCOMPUTE_CAPABILITY=$(opticks-compute-capability) \
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

opticks-configure()
{
   opticks-wipe

   case $(opticks-cmake-generator) in
       "Visual Studio 14 2015") opticks-configure-local-boost $* ;;
                             *) opticks-configure-system-boost $* ;;
   esac
}

opticks-configure-system-boost()
{
   opticks-cmake $* 
}

opticks-configure-local-boost()
{
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


opticks-ctest-()
{
   [ "$(which ctest 2>/dev/null)" == "" ] && ctest3 $* || ctest $* 
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
   if [ -d "$bdir" ]; then
       shift
   else
       bdir=$(opticks-bdir) 
   fi

   cd $bdir

   opticks-ctest- $*

   cd $iwd
   echo $msg use -V to show output 
}


opticks-find(){
   local str=${1:-ENV_HOME}

   local iwd=$PWD
   opticks-scd

   find . -name '*.cc' -exec grep -H $str {} \;
   find . -name '*.hh' -exec grep -H $str {} \;
   find . -name '*.cpp' -exec grep -H $str {} \;
   find . -name '*.hpp' -exec grep -H $str {} \;
   find . -name '*.h' -exec grep -H $str {} \;

   cd $iwd
}


opticks-unset--()
{
   local pfx=${1:-OPTICKS_}
   local kv
   local k
   local v
   env | grep $pfx | while read kv ; do 

       k=${kv/=*}
       v=${kv/*=}

       #printf "%50s %s \n" $k $v  
       echo unset $k 
   done
}

opticks-unset-()
{
   opticks-unset-- OPTICKS_
   opticks-unset-- DAE_
   opticks-unset-- IDPATH
}
opticks-unset()
{
   local tmp=/tmp/unset.sh
   opticks-unset- >  $tmp

   echo unset with : . $tmp
}




opticks-all-projs-(){ cat << EOP
sysrap
brap
npy
okc
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

ggeoview
cfg4
EOP
}


opticks-cuda-projs-(){ cat << EOP
cudarap
thrustrap
optixrap
opticksop
opticksgl
EOP
}


opticks---(){ 
   local arg=${1:-all}
   local proj
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      $proj-
      $proj--
   done
} 

opticks----(){ 
   ## proj--- touches the API header and then does $proj-- : this forcing recompilation of everything 
   local arg=${1:-all}
   local proj
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      $proj-
      echo proj $proj
      $proj---
   done
} 

opticks-list()
{
   local arg=${1:-all}
   local proj
   opticks-${arg}-projs- | while read proj ; do
      [ -z "$proj" ] && continue  
      echo proj $proj
   done
}




opticks-nuclear(){   rm -rf $LOCAL_BASE/opticks/* ; }
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


### opticks projs ###  **moved** all projs into top level folders

sysrap-(){          . $(opticks-home)/sysrap/sysrap.bash && sysrap-env $* ; }
brap-(){            . $(opticks-home)/boostrap/brap.bash && brap-env $* ; }
npy-(){             . $(opticks-home)/opticksnpy/npy.bash && npy-env $* ; }
okc-(){             . $(opticks-home)/optickscore/okc.bash && okc-env $* ; }

ggeo-(){            . $(opticks-home)/ggeo/ggeo.bash && ggeo-env $* ; }
assimprap-(){       . $(opticks-home)/assimprap/assimprap.bash && assimprap-env $* ; }
openmeshrap-(){     . $(opticks-home)/openmeshrap/openmeshrap.bash && openmeshrap-env $* ; }
opticksgeo-(){      . $(opticks-home)/opticksgeo/opticksgeo.bash && opticksgeo-env $* ; }

oglrap-(){          . $(opticks-home)/oglrap/oglrap.bash && oglrap-env $* ; }
cudarap-(){         . $(opticks-home)/cudarap/cudarap.bash && cudarap-env $* ; }
thrustrap-(){       . $(opticks-home)/thrustrap/thrustrap.bash && thrustrap-env $* ; }
optixrap-(){        . $(opticks-home)/optixrap/optixrap.bash && optixrap-env $* ; }

opticksop-(){       . $(opticks-home)/opticksop/opticksop.bash && opticksop-env $* ; }
opticksgl-(){       . $(opticks-home)/opticksgl/opticksgl.bash && opticksgl-env $* ; }
ggeoview-(){        . $(opticks-home)/ggeoview/ggeoview.bash && ggeoview-env $* ; }
cfg4-(){            . $(opticks-home)/cfg4/cfg4.bash && cfg4-env $* ; }
ana-(){             . $(opticks-home)/ana/ana.bash && ana-env $*  ; }

### opticks launchers ########

oks-(){             . $(opticks-home)/bin/oks.bash && oks-env $* ; }
ggv-(){             . $(opticks-home)/bin/ggv.bash && ggv-env $* ; }
vids-(){            . $(opticks-home)/bin/vids.bash && vids-env $* ; }
op-(){              . $(opticks-home)/bin/op.sh ; }

