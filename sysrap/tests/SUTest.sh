#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=SUTest
dir=/tmp/$name
mkdir -p $dir

nvcc -c ../SU.cu -I.. -o $dir/SU.o
[ $? -ne 0 ] && echo $msg nvcc compile error && exit 1 

CUDA_PREFIX=/usr/local/cuda
gcc $name.cc $dir/SU.o -std=c++11 -lstdc++ -I.. -I$CUDA_PREFIX/include -L$CUDA_PREFIX/lib -lcudart -o $dir/$name
[ $? -ne 0 ] && echo $msg gcc compile error && exit 2

$dir/$name
[ $? -ne 0 ] && echo $msg run error && exit 3

exit 0  


