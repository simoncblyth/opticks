#!/bin/bash -l 

name=scerenkov__test
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

gcc $name.cc -std=c++11 -I.. -I$CUDA_PREFIX/include  -lstdc++ -Wall -fstrict-aliasing -Wstrict-aliasing -O3  -o $bin 
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0

