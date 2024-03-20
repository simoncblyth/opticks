#!/bin/bash -l 
usage(){ cat << EOU
SCE_test.sh
=============

::

   ~/o/sysrap/tests/SCE_test.sh 

EOU
}

name=SCE_test
bin=/tmp/$name

cd $(dirname $(realpath $BASH_SOURCE)) 

glm-

cuda_prefix=/usr/local/cuda 
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

gcc $name.cc \
    -std=c++11 -lstdc++ \
     -I..  \
     -I$(glm-prefix) \
     -I$CUDA_PREFIX/include \
     -o $bin \
     && $bin 



