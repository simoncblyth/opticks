#!/usr/bin/env bash

usage(){ cat << EOU
SKernelProfile_test.sh
========================

~/o/sysrap/tests/SKernelProfile_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SKernelProfile_test


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

test=ctor
export TEST=${TEST:-$test}


defarg="info_nvcc_gcc_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name tmp TMP FOLD bin defarg arg cuda_prefix CUDA_PREFIX test TEST"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/nvcc}" != "$arg" ]; then
   nvcc -c ../SU.cu -I.. -o $FOLD/SU.o
   [ $? -ne 0 ] && echo $FUNCNAME nvcc compile error && exit 1
fi

if [ "${arg/gcc}" != "$arg" ]; then
   gcc $name.cc $FOLD/SU.o -std=c++17 -lstdc++ -I.. -I$CUDA_PREFIX/include -L$CUDA_PREFIX/lib64 -lcudart -lm  -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE gcc compile error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 4
fi



