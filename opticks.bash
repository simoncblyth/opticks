# === func-gen- : opticks fgp ./opticks.bash fgn opticks fgh .
opticks-(){         source $(opticks-source) ; }
opticks-src(){      echo opticks.bash ; }
opticks-source(){   echo ${BASH_SOURCE:-$(env-home)/$(opticks-src)} ; }
opticks-vi(){       vi $(opticks-source) ; }
opticks-env(){      elocal- ; }
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

    simon:~ blyth$ rm -rf /usr/local/opticks/* ; opticks- ; opticks--


TODO
-----

* standardize names: Cfg Bcfg bcfg- cfg-
* tidy up ssl and crypto : maybe in NPY_LIBRARIES 
* tidy up optix optixu FindOptiX from the SDK doesnt set OPTIX_LIBRARIES

* get NPY tests to pass
* role out CTest-ing to all packages, get the tests to pass 


* check OptiX 4.0 beta for cmake changes 
* externalize or somehow exclude from standard building the Rap pkgs, as fairly stable
* look into isolating Assimp dependency usage
* machinery for getting externals
* spawn opticks repository 
* adopt single level directories 
* split ggv- usage from ggeoview- building

* investigate CPack, CTest



Dependencies of internals
---------------------------

::

   =====================  ===============  =============   ==============================================================================
   directory              precursor        pkg name        required find package 
   =====================  ===============  =============   ==============================================================================
   boost/bpo/bcfg         bcfg-            Cfg             Boost
   boost/bregex           bregex-          Bregex          Boost
   graphics/ppm           ppm-             PPM             
   numerics/npy           npy-             NPY             Boost GLM Bregex 
   optickscore            optickscore-     OpticksCore     Boost GLM Bregex Cfg NPY 
   optix/ggeo             ggeo-            GGeo            Boost GLM Bregex Cfg NPY OpticksCore
   graphics/assimprap     assimprap-       AssimpRap       Boost Assimp GGeo GLM NPY OpticksCore
   graphics/openmeshrap   openmeshrap-     OpenMeshRap     Boost GLM NPY GGeo OpticksCore OpenMesh 
   graphics/oglrap        oglrap-          OGLRap          GLEW GLFW GLM Boost Cfg Opticks GGeo PPM NPY Bregex ImGui        
   cuda/cudarap           cudarap-         CUDARap         CUDA (ssl)
   numerics/thrustrap     thrustrap-       ThrustRap       CUDA Boost GLM NPY CUDARap 
   graphics/optixrap      optixrap-        OptiXRap        OptiX CUDA Boost GLM NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap 
   opticksop              opticksop-       OpticksOp       OptiX CUDA Boost GLM Cfg Opticks GGeo NPY OptiXRap CUDARap ThrustRap      
   opticksgl              opticksgl-       OpticksGL       OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY OpticksCore Assimp AssimpRap GGeo CUDARap ThrustRap OptiXRap OpticksOp
   graphics/ggeoview      ggv-             GGeoView        OptiX CUDA Boost GLM GLEW GLFW OGLRap NPY Cfg OpticksCore 
                                                           Assimp AssimpRap OpenMesh OpenMeshRap GGeo ImGui Bregex OptiXRap CUDARap ThrustRap OpticksOp OpticksGL 
   optix/cfg4             cfg4-            CfG4            Boost Bregex GLM NPY Cfg GGeo OpticksCore Geant4 EnvXercesC G4DAE 
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
EnvXercesC
G4DAE
Assimp
OpenMesh
GLEW
GLEQ
GLFW
ImGui
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


opticks-dir(){ echo $(local-base)/opticks ; }
opticks-cd(){  cd $(opticks-dir); }


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

opticks-txt-list(){
  local dir
  opticks-dirs | while read dir 
  do
      echo $dir/CMakeLists.txt
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
       -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
       $(opticks-sdir)

   cd $iwd
}

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
