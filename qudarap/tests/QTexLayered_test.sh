#!/bin/bash

usage(){ cat << EOU
QTexLayered_test.sh
=====================

Test creation of layered texture from NP.hh array.

~/o/qudarap/tests/QTexLayered_test.sh


EOU
}

name=QTexLayered_test

cd $(dirname $(realpath $BASH_SOURCE))

defarg="info_gcc_run"
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

if [ "${arg/gcc}" != "$arg" ]; then
    gcc $name.cc \
         -g -std=c++17 -lstdc++ -lcudart \
         -I.. \
         -DWITH_CUDA \
         -I../../sysrap \
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

