#!/bin/bash -l 
usage(){ cat << EOU
SGLFW_SOPTIX_Scene_test.sh
=============================

Usage and impl::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.cc
    
Related simpler priors::

    ~/o/sysrap/tests/SOPTIX_Scene_test.sh
    ~/o/sysrap/tests/SOPTIX_Scene_test.cc

Preqs::

    ~/o/sysrap/tests/SScene_test.sh 
        ## create and persist SScene.h from loaded stree.h
  
    ~/o/sysrap/tests/SScene_test.sh 
        ## create and persist stree.h (eg from loaded gdml via GEOM config)


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
name=SGLFW_SOPTIX_Scene_test

export FOLD=/tmp/$name
bin=$FOLD/$name
mkdir -p $FOLD


cu=../SOPTIX.cu
ptx=$FOLD/SOPTIX.ptx
export SOPTIX_PTX=$ptx 
# when using CMake generated ptx will be smth like:$OPTICKS_PREFIX/ptx/sysrap_generated_SOPTIX.cu.ptx 
# following pattern $OPTICKS_PREFIX/ptx/CSGOptiX_generated_CSGOptiX7.cu.ptx" 

opticks-
glm-

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done 

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}


scene=0
case ${SCENE:-$scene} in 
0) scene_fold=/tmp/SScene_test ;;
1) scene_fold=$HOME/.opticks/GEOM/RaindropRockAirWater/CSGFoundry/SSim ;;
esac
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}


shader_fold=../../examples/UseShaderSGLFW_SScene_encapsulated/gl
export SHADER_FOLD=${SHADER_FOLD:-$shader_fold}

dump=1
DUMP=${DUMP:-$dump}
export SGLM__set_frame_DUMP=$DUMP

#export SGLFW_SOPTIX_Scene_test_DUMP=1  


wh=1024,768
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
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export TMIN=${TMIN:-$tmin}
export ESCALE=${ESCALE:-$escale}
export CAM=${CAM:-$cam}
export FULLSCREEN=${FULLSCREEN:-$fullscreen}


handle=-1 # -1:IAS 0...8 GAS indices 
export HANDLE=${HANDLE:-$handle}
:
frame=-1
export FRAME=${FRAME:-$frame}

vizmask=t   # 0xff no masking
#vizmask=t0  # 0xfe mask global
export VIZMASK=${VIZMASK:-$vizmask}


defarg="info_ptx_build_run"
arg=${1:-$defarg}


if [ ! -d "$SCENE_FOLD/scene" ]; then
  echo $BASH_SOURCE : FATAL SCENE_FOLD $SCENE_FOLD does not contain scene 
  arg=info  
fi 



PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE CUDA_PREFIX OPTIX_PREFIX cuda_l SCENE_FOLD FOLD SOPTIX_PTX bin"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 

if [ "${arg/ptx}" != "$arg" ]; then
    echo $BASH_SOURCE ptx
    nvcc $cu \
        -ptx -std=c++11 \
        -c \
        -lineinfo \
        -use_fast_math \
        -I.. \
        -I$CUDA_PREFIX/include  \
        -I$OPTIX_PREFIX/include  \
        -o $ptx
    [ $? -ne 0 ] && echo $BASH_SOURCE : ptx build error && exit 1 
    echo $BASH_SOURCE ptx DONE
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
        -g -O0 -std=c++11 \
        -I$SYSRAP_DIR \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I$OPTICKS_PREFIX/externals/include \
        -I$CUDA_PREFIX/include \
        -I$OPTIX_PREFIX/include \
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

