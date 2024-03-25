#!/bin/bash -l 
usage(){ cat << EOU
SOPTIX_Scene_test.sh
=====================

::

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
name=SOPTIX_Scene_test

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
    gcc $name.cc \
        -std=c++11 -lstdc++ -lm -ldl  -g \
        -I${SYSRAP_DIR} \
        -I$CUDA_PREFIX/include \
        -I$OPTIX_PREFIX/include \
        -I$(glm-prefix) \
        -L$CUDA_PREFIX/$cuda_l -lcudart \
        -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi



exit 0 

