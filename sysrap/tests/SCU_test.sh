#!/bin/bash 
usage(){ cat << EOU

~/o/sysrap/tests/SCU_test.sh 

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))


defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

name=SCU_test
bin=/tmp/$name

gcc $name.cc \
    -I$CUDA_PREFIX/include \
    -I.. \
    -L$CUDA_PREFIX/lib64 \
    -std=c++17 -lstdc++ -lcudart -g -o $bin && $bin



