# === func-gen- : opticks fgp ./opticks.bash fgn opticks fgh .
opticks-(){         source $(opticks-source) ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-usage(){ cat << EOU

opticks : experiment with umbrella cmake building
====================================================

Aiming for this to go in top level of a new Opticks repo
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
    simon:~ blyth$ opticks- ; opticks--


TODO
-----

* tidy up ssl and crypto : maybe in NPY_LIBRARIES 
* tidy up optix optixu FindOptiX from the SDK doesnt set OPTIX_LIBRARIES

* role out CTest-ing to all packages, get the tests to pass 


* incorporate cfg4- in superbuild with G4 checking

* check OptiX 4.0 beta for cmake changes 
* externalize or somehow exclude from standard building the Rap pkgs, as fairly stable
* look into isolating Assimp dependency usage
* machinery for getting externals
* spawn opticks repository 
* adopt single level directories 
* split ggv- usage from ggeoview- building

* investigate CPack, CTest


Opticks internals table with dependencies 
--------------------------------------------


=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        required find package 
=====================  ===============  =============   ==============================================================================
boost/bpo/bcfg         bcfg-            BCfg            Boost
boost/bregex           bregex-          BRegex          Boost
graphics/ppm           ppm-             PPM             
numerics/npy           npy-             NPY             Boost GLM BRegex 
optickscore            optickscore-     OpticksCore     Boost GLM BRegex BCfg NPY 
optix/ggeo             ggeo-            GGeo            Boost GLM BRegex BCfg NPY OpticksCore
graphics/assimprap     assimprap-       AssimpRap       Boost Assimp GGeo GLM NPY OpticksCore
graphics/openmeshrap   openmeshrap-     OpenMeshRap     Boost GLM NPY GGeo OpticksCore OpenMesh 
graphics/oglrap        oglrap-          OGLRap          GLEW GLFW GLM Boost BCfg Opticks GGeo PPM NPY BRegex ImGui        
cuda/cudarap           cudarap-         CUDARap         CUDA (ssl)
numerics/thrustrap     thrustrap-       ThrustRap       CUDA Boost GLM NPY CUDARap 
graphics/optixrap      optixrap-        OptiXRap        OptiX CUDA Boost GLM NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap 
opticksop              opticksop-       OpticksOp       OptiX CUDA Boost GLM BCfg Opticks GGeo NPY OptiXRap CUDARap ThrustRap      
opticksgl              opticksgl-       OpticksGL       OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap OptiXRap OpticksOp
graphics/ggeoview      ggv-             GGeoView        OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY BCfg OpticksCore 
                                                       Assimp AssimpRap OpenMesh OpenMeshRap GGeo ImGui BRegex OptiXRap CUDARap ThrustRap OpticksOp OpticksGL 
optix/cfg4             cfg4-            CfG4            Boost BRegex GLM NPY BCfg GGeo OpticksCore Geant4 EnvXercesC G4DAE 
=====================  ===============  =============   ==============================================================================

* ppm-/loadPPM.h now privately in oglrap-



Externals 
-----------

Infrastructure 
~~~~~~~~~~~~~~~~

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
boost                  boost-           Boost           using macports: system thread program_options log log_setup filesystem regex 
=====================  ===============  =============   ==============================================================================

Geometry
~~~~~~~~~~~

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
graphics/assimp        assimp-          Assimp          using github fork of assimp incoporating handling of G4DAE extras 
graphics/openmesh      openmesh-        OpenMesh        
=====================  ===============  =============   ==============================================================================

OpenGL related
~~~~~~~~~~~~~~~

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
graphics/glm           glm-             GLM             header only
graphics/glew          glew-            GLEW            OpenGL extensions loading library   
graphics/glfw          glfw-            GLFW            library for creating windows with OpenGL and receiving input   
graphics/gleq          gleq-            GLEQ            GLFW authors example addition of event style input handling 
graphics/gui/imgui     imgui-                           uncontrolled version: git clone https://github.com/ocornut/imgui.git
=====================  ===============  =============   ==============================================================================

* TODO: clone imgui into my github account and grab from there, so the version is fixed


CUDA related
~~~~~~~~~~~~~

=====================  ===============  =============   ==============================================================================
directory              precursor        pkg name        notes
=====================  ===============  =============   ==============================================================================
cuda                   cuda-            CUDA
numerics/thrust        thrust-          Thrust
optix                  optix-           OptiX
=====================  ===============  =============   ==============================================================================


Usage
-------

::

   . opticks.bash 

   opticks-cmake
   opticks-install

   opticks-run


Pristine cycle::

   e;. opticks.bash;opticks-wipe;opticks-cmake;opticks-install


To Consider
-------------

* conditional cfg4 build depending on finding G4 

* externals depend on env bash functions for getting and installing
  and env cmake modules for finding 

* externals gathering


ggv.sh Launcher
-----------------

Bash launcher ggv.sh tied into the individual bash functions and 
sets up envvars::

   OPTICKS_GEOKEY
   OPTICKS_QUERY
   OPTICKS_CTRL
   OPTICKS_MESHFIX
   OPTICKS_MESHFIX_CFG

* OpticksResource looks for a metadata sidecar .ini accompanying the .dae
  eg for /tmp/g4_00.dae the file /tmp/g4_00.ini is looked for

* TODO: enable all envvars to come in via the metadata .ini approach with 
  potential to be overridded by the envvars 


Umbrella CMakeLists.txt
-------------------------

* avoid tests in different pkgs with same name 

Thoughts
--------

The umbrella cmake build avoids using the bash functions
for each of the packages... but those are kinda useful
for development. 

EOU
}

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
Cfg
Bregex
PPM
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
GLM

Assimp
OpenMesh
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
  local base=$ENV_HOME/CMake/Modules
  opticks-${1} | while read f 
  do
     echo $base/Find${f}.cmake
  done 
}

opticks-ifind(){ vi $(opticks-tfind- internals) ; }
opticks-xfind(){ vi $(opticks-tfind- xternals) ; }
opticks-ofind(){ vi $(opticks-tfind- other) ; }

opticks-distclean(){
   local names="bin build gl include lib ptx"
   local base=$(opticks-dir)
   local name
   local msg="# $FUNCNAME : "
   echo $msg pipe to sh to do the deletion
   for name in $names 
   do 
      local dir=$base/$name
      [ -d "$dir" ] && echo rm -rf $dir ;
   done
}

opticks-dir(){ echo $(local-base)/opticks ; }
opticks-cd(){  cd $(opticks-dir); }

opticks-prefix(){ echo $(local-base)/opticks ; }
opticks-home(){   echo $(env-home) ; }

opticks-env(){      
   elocal- 
   glfw-
}


opticks-cmake(){
   local msg="=== $FUNCNAME : "
   local iwd=$PWD

   local bdir=$(opticks-bdir)
   mkdir -p $bdir

   [ ! -d "$bdir" ] && echo $msg NO bdir $bdir && return  

   opticks-bcd
   cmake \
       -DWITH_OPTIX:BOOL=ON \
       -DCMAKE_BUILD_TYPE=Debug \
       -DCMAKE_INSTALL_PREFIX=$(opticks-idir) \
       -DCMAKE_INSTALL_PREFIX=$(opticks-idir) \
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       $(opticks-sdir)

   cd $iwd
}


opticks-externals-install(){
   local msg="=== $FUNCNAME :"

   local exts="glm glfw glew gleq"

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




opticks-sdir(){ echo $(env-home) ; }
opticks-idir(){ echo $(local-base)/opticks ; }
opticks-bdir(){ echo $(local-base)/opticks/build ; }
opticks-tdir(){ echo /tmp/opticks ; }

opticks-scd(){  cd $(opticks-sdir); }
opticks-cd(){   cd $(opticks-sdir); }
opticks-icd(){  cd $(opticks-idir); }
opticks-bcd(){  cd $(opticks-bdir); }


opticks-txt(){   cd $ENV_HOME ; vi CMakeLists.txt $(opticks-txt-list) ; }
opticks-bash(){  cd $ENV_HOME ; vi opticks.bash $(opticks-bash-list) ; }
opticks-edit(){  cd $ENV_HOME ; vi opticks.bash $(opticks-bash-list) CMakeLists.txt $(opticks-txt-list) ; } 
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

opticks-wipe(){
   local bdir=$(opticks-bdir)
   rm -rf $bdir
}

opticks-optix-install-dir(){ echo /Developer/OptiX ; }



opticks-bin(){ echo $(opticks-idir)/bin/GGeoView ; }


opticks-make(){
   local iwd=$PWD

   opticks-bcd
   make $*

   cd $iwd
}

opticks-install(){
   opticks-make install
}


opticks--()
{
    local msg="$FUNCNAME :"
    echo $msg START $(date)

    opticks-wipe
    opticks-cmake
    #opticks-make
    opticks-install
    # -install seems to duplicate -make ? maybe can do with just -install

    echo $msg DONE $(date)
}

opticks-run()
{

    export-
    export-export   ## needed to setup DAE_NAME_DYB the envvar name pointed at by the default opticks_GEOKEY 

    local bin=$(opticks-bin)
    $bin $*         ## bare running with no bash script, for checking defaults 

}
