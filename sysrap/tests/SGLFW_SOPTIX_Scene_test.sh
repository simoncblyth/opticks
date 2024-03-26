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

scene_fold=/tmp/SScene_test
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}

shader_fold=../../examples/UseShaderSGLFW_SScene_encapsulated/gl
export SHADER_FOLD=${SHADER_FOLD:-$shader_fold}


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

#cam=perspective
cam=orthographic

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


handle=-1 # -1:IAS 0...8 GAS indices 
export HANDLE=${HANDLE:-$handle}




defarg="info_ptx_build_run"
arg=${1:-$defarg}

PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE CUDA_PREFIX OPTIX_PREFIX cuda_l SCENE_FOLD FOLD SOPTIX_PTX"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 

if [ "${arg/ptx}" != "$arg" ]; then
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
fi

if [ "${arg/build}" != "$arg" ]; then

    [ "$(uname)" == "Darwin" ] && echo $BASH_SOURCE : ERROR : THIS NEEDS OPTIX7+ SO LINUX ONLY && exit 1

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

