#!/bin/bash
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

#name=SOPTIX_Scene_test
name=SOPTIX_Scene_Encapsulated_test

export FOLD=/tmp/$name
bin=$FOLD/$name
mkdir -p $FOLD

export PPM_PATH=$FOLD/$name.ppm

cu=../SOPTIX.cu
ptx=$FOLD/SOPTIX.ptx
export SOPTIX_PTX=$ptx 


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done 

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}

if [ -z "$OPTIX_PREFIX" ]; then 
   echo $0 - MISSING OPTIX_PREFIX && exit 1 
fi 


sysrap_dir=..
SYSRAP_DIR=${SYSRAP_DIR:-$sysrap_dir}

scene_fold=/tmp/SScene_test
export SCENE_FOLD=${SCENE_FOLD:-$scene_fold}



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




defarg="info_ptx_build_run_open"
arg=${1:-$defarg}

PATH=$PATH:$CUDA_PREFIX/bin

vars="BASH_SOURCE CUDA_PREFIX OPTIX_PREFIX OPTICKS_PREFIX cuda_l SCENE_FOLD FOLD SOPTIX_PTX"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 

if [ "${arg/ptx}" != "$arg" ]; then
   nvcc $cu \
        -ptx -std=c++17 \
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
        -std=c++17 -lstdc++ -lm -ldl  -g \
        -I${SYSRAP_DIR} \
        -I$CUDA_PREFIX/include \
        -I$OPTIX_PREFIX/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
         -DWITH_CHILD \
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
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/open}" != "$arg" ]; then
    [ -z "$DISPLAY" ] && echo $BASH_SOURCE adhoc setting DISPLAY && export DISPLAY=:0 
    open $PPM_PATH
    [ $? -ne 0 ] && echo $BASH_SOURCE : open error && exit 4
fi


exit 0 

