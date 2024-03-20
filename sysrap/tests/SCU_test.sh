#!/bin/bash -l 
usage(){ cat << EOU
SCU_test.sh
============

::

    ~/o/sysrap/tests/SCU_test.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SCU_test
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

gcc $name.cc \
     -std=c++11 -lstdc++ -g \
     -I$CUDA_PREFIX/include \
     -I.. \
     -L$CUDA_PREFIX/lib -lcudart \
     -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

exit 0 

