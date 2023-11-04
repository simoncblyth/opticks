#!/bin/bash -l 

cd $(dirname $BASH_SOURCE)

name=saabb_Test

FOLD=/tmp/$USER/opticks/$name
bin=$FOLD/$name

mkdir -p $FOLD

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

gcc $name.cc \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -std=c++11 \
    -lstdc++ \
    -lm \
    -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE : compile fail && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run fail && exit 2

exit 0 


