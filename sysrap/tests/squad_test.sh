#!/bin/bash -l 

name=squad_test
bin=/tmp/$name

gcc $name.cc \
     -I.. \
     -I${CUDA_PREFIX:-/usr/local/cuda}/include \
     -std=c++11 \
     -lstdc++ \
     -o /tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 

exit 0 

