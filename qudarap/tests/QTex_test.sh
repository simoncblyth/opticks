#!/bin/bash

usage(){ cat << EOU


~/o/qudarap/tests/QTex_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=QTex_test

defarg="info_build_run"
arg=${1:-$defarg}
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE arg name FOLD CUDA_PREFIX OPTICKS_PREFIX"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then

    #-DMOCK_TEXTURE \
    gcc $name.cc ../QTex.cc \
         -g -std=c++17 -lstdc++ -lcudart \
         -DWITH_CUDA \
         -I.. \
         -I$OPTICKS_PREFIX/include/SysRap \
         -I$CUDA_PREFIX/include \
         -L$CUDA_PREFIX/lib64 \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : eun error && exit 2
fi



exit 0
