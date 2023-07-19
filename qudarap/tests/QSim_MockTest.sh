#!/bin/bash -l 
usage(){ cat << EOU
QSim_MockTest.sh
==================

Note that unlike the standard Opticks CMake build
this script finds Custom4 without consulting 
the CMAKE_PREFIX_PATH 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=QSim_MockTest

source $HOME/.opticks/GEOM/GEOM.sh 

defarg="build_run"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

custom4_prefix=${OPTICKS_PREFIX}_externals/custom4/0.1.6
CUSTOM4_PREFIX=${CUSTOM4_PREFIX:-$custom4_prefix}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="FOLD GEOM bin name CUSTOM4_PREFIX CUDA_PREFIX"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/check}" != "$arg" ]; then
    path="${CUSTOM4_PREFIX}/include/Custom4/C4CustomART.h"
    if [ -f "$path" ]; then
       echo $BASH_SOURCE : path $path : EXISTS
    else
       echo $BASH_SOURCE : path $path : DOES NOT EXIST
    fi 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       ../QPMT.cc \
       ../QOptical.cc \
       ../QBnd.cc \
       ../QTex.cc \
       ../QProp.cc \
       ../QBase.cc \
       -g \
       -std=c++11 -lstdc++ \
       -DDEBUG_PIDX \
       -DMOCK_CURAND \
       -DMOCK_CURAND_DEBUG \
       -DMOCK_TEXTURE \
       -I.. \
       -I$OPTICKS_PREFIX/include/SysRap  \
       -I$CUDA_PREFIX/include \
       -I$OPTICKS_PREFIX/externals/glm/glm \
       -I$OPTICKS_PREFIX/externals/plog/include \
       -DWITH_CUSTOM4 -I$CUSTOM4_PREFIX/include/Custom4 \
       -o $bin 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 


if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $msg dbg error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py
    [ $? -ne 0 ] && echo $msg ana error && exit 4
fi


exit 0 



