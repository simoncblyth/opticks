#!/bin/bash 
usage(){ cat << EOU
SGLFW_Event_test.sh : triangulated raytrace and rasterized visualization
=================================================================================

Assuming the scene folder exists already::

    ~/o/sysrap/tests/SGLFW_Event_test.sh
    SCENE=0 ~/o/sysrap/tests/SGLFW_Event_test.sh
    SCENE=1 ~/o/sysrap/tests/SGLFW_Event_test.sh
    SCENE=2 ~/o/sysrap/tests/SGLFW_Event_test.sh
    SCENE=3 ~/o/sysrap/tests/SGLFW_Event_test.sh
    ## SCENE picks between different scene directories

Impl::

    ~/o/sysrap/tests/SGLFW_Event_test.cc


One step way to create the stree and scene from gdml or other geometry source
---------------------------------------------------------------------------------

::

    ~/o/u4/tests/U4TreeCreateSSimTest.sh  


Two step way to create the scene folder from gdml or other geometry source 
-----------------------------------------------------------------------------

1. load geometry and use U4Tree::Create to convert to stree.h and persist the tree::   

    ~/o/u4/tests/U4TreeCreateTest.sh 
        # create and persist stree.h (eg from gdml or j/PMTSim via GEOM config)

2. load stree and create corresponding SScene::

    ~/o/sysrap/tests/SScene_test.sh 
        # create and persist SScene.h from loaded stree.h
  

Simpler Prior Developments
---------------------------
    
Related simpler priors::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc



Which GL/glew.h is picked up ?
---------------------------------

On changing env from JUNO-opticks to ONLY-opticks build directly with 
opticks-full note that have changed the OpenGL version in use 
causing missing symbol GL_CONTEXT_LOST

Fixed that with GL_VERSION_4_5 check, but why the older version ?

* probably just a change in glew header not a change in actual GL version used

Adding "-M" to gcc commandline lists all the 
included headers in Makefile dependency format.
This shows are picking up system headers::

    /usr/include/GL/glew.h
    /usr/include/GL/glu.h
    /usr/include/GL/gl.h
    /usr/include/GLFW/glfw3.h

Which corresponds to::

    //SGLFW::init GL_RENDERER [NVIDIA TITAN RTX/PCIe/SSE2] 
    //SGLFW::init GL_VERSION [4.1.0 NVIDIA 515.43.04] 

HMM: this maybe because I removed glew and glfw from the 
standard externals ?

HMM: but after installing those get::

   undefined symbol: glfwSetWindowMaximizeCallback

when using::

    /data/blyth/opticks_Debug/externals/include/GL/glew.h
    /usr/include/GL/glu.h
    /usr/include/GL/gl.h
    /data/blyth/opticks_Debug/externals/include/GLFW/glfw3.h

Return to system with::

   glfw-
   glfw-manifest-wipe
   glfw-manifest-wipe | sh 

   glew-
   glew-manifest-wipe
   glew-manifest-wipe | sh 



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SGLFW_Event_test


#source $HOME/.opticks/GEOM/GEOM.sh
#[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1 


export FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name
mkdir -p $FOLD


# when using CMake generated ptx will be smth like:$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx 
# following pattern $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done 

#optix_prefix=${OPTICKS_OPTIX_PREFIX}
#OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}

 
scene=0
case ${SCENE:-$scene} in 
0) scene_fold=$HOME/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/SSim;;
1) scene_fold=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim ;;
2) scene_fold=$TMP/G4CXOpticks_setGeometry_Test/$GEOM/CSGFoundry/SSim ;;
3) scene_fold=$TMP/U4TreeCreateSSimTest/$GEOM/SSim ;;
esac
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}


shader_fold=../../examples/UseShaderSGLFW_SScene_encapsulated/gl
export SHADER_FOLD=${SHADER_FOLD:-$shader_fold}

export RECORDER_SHADER_FOLD=../../examples/UseGeometryShader/rec_flying_point/
dump=0
DUMP=${DUMP:-$dump}
export SGLM__set_frame_DUMP=$DUMP

export SGLFW_Event_test_DUMP=1  


#wh=1024,768
wh=2560,1440

#eye=0.1,0,-10
#eye=-1,-1,0
#eye=-10,-10,0
#eye=-10,0,0
#eye=0,-10,0
eye=-1,-1,0
up=0,0,1
look=0,0,0

cam=perspective
#cam=orthographic

#fullscreen=0
fullscreen=1

tmin=0.1    
#escale=asis
escale=extent


export WH=${WH:-$wh}
export FULLSCREEN=${FULLSCREEN:-$fullscreen}
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export TMIN=${TMIN:-$tmin}
export ESCALE=${ESCALE:-$escale}
export CAM=${CAM:-$cam}


handle=-1 # -1:IAS 0...8 GAS indices 
export HANDLE=${HANDLE:-$handle}
:
frame=-1
export SGLFW_FRAME=${SGLFW_FRAME:-$frame}

vizmask=t   # 0xff no masking
#vizmask=t0  # 0xfe mask global
export VIZMASK=${VIZMASK:-$vizmask}


#export SOPTIX_SBT__initHitgroup_DUMP=1

export SRECORD_PATH=/tmp/sphoton_test/record.npy


defarg="info_build_run"
arg=${1:-$defarg}


if [ ! -d "$SCENE_FOLD/scene" ]; then
  echo $BASH_SOURCE : FATAL SCENE_FOLD $SCENE_FOLD does not contain scene
  echo $BASH_SOURCE : with newly created CSGFoundry/SSim there is no need for manual SCENE_FOLD as will be in CSGFoundry/SSim/scene  
  #arg=info  
fi 



PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE defarg arg CUDA_PREFIX cuda_l SCENE_FOLD FOLD bin SGLFW_FRAME"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 


if [ "${arg/build}" != "$arg" ]; then

    echo $BASH_SOURCE build
    [ "$(uname)" == "Darwin" ] && echo $BASH_SOURCE : ERROR : THIS NEEDS OPTIX7+ SO LINUX ONLY && exit 1

    # -M lists paths of all included headers in Makefile dependency format 
    # -M \
    gcc $name.cc \
        -fvisibility=hidden \
        -fvisibility-inlines-hidden \
        -fdiagnostics-show-option \
        -Wall \
        -Wno-unused-function \
        -Wno-shadow \
        -Wsign-compare \
        -DWITH_CUDA_GL_INTEROP \
        -DWITH_CHILD \
        -g -O0 -std=c++11 \
        -I$SYSRAP_DIR \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -L$CUDA_PREFIX/$cuda_l -lcudart \
        -lstdc++ \
        -lm -ldl \
        -L$OPTICKS_PREFIX/externals/lib -lGLEW \
        -L$OPTICKS_PREFIX/externals/lib64 -lglfw \
        -lGL  \
        -o $bin

 
        
    # -Wno-unused-private-field \  ## clang-ism ? 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
    echo $BASH_SOURCE build DONE
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then

    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$OPTICKS_PREFIX/externals/lib:$OPTICKS_PREFIX/externals/lib64
    echo $BASH_SOURCE : Linux running $bin : with some manual LD_LIBRARY_PATH config 

    [ -z "$DISPLAY" ] && echo $BASH_SOURCE adhoc setting DISPLAY && export DISPLAY=:0 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi


exit 0 

