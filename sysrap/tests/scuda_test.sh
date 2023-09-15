#!/bin/bash -l 

name=scuda_test 
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

gcc $name.cc \
      -std=c++11 -lstdc++ \
       -I.. \
       -I${CUDA_PREFIX}/include \
       -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 

