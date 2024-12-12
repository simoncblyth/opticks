#!/bin/bash 
usage(){ cat << EOU

~/o/sysrap/tests/srng_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

name=srng_test
bin=/tmp/$name

gcc $name.cc \
    -DWITH_CURANDLITE \
    -I$CUDA_PREFIX/include \
    -I.. \
    -std=c++17 -lstdc++ -g -o $bin && $bin

