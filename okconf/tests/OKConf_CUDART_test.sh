#!/bin/bash
usage(){ cat << EOU
OKConf_CUDART_test.sh
=======================

~/o/okconf/tests/OKConf_CUDART_test.sh

Testing the header-only OKConf_CUDART.h

EOU
}

name=OKConf_CUDART_test
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


cd $(dirname $(realpath $BASH_SOURCE))


gcc $name.cc \
      -std=c++17 -lstdc++ \
      -I.. \
      -I$CUDA_PREFIX/include \
      -L$CUDA_PREFIX/lib64 \
      -lcudart \
      -DWITH_CUDA \
      -I$OPTICKS_PREFIX/include/OKConf \
      -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0


