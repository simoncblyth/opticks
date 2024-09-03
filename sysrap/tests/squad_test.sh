#!/bin/bash -l 
usage(){ cat << EOU
quad_test.sh
=============

~/o/sysrap/tests/squad_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=squad_test
bin=/tmp/$name


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


gcc $name.cc \
     -I.. \
     -I${CUDA_PREFIX}/include \
     -std=c++11 \
     -lstdc++ -lm \
     -o $bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 

exit 0 

