#!/bin/bash -l 
usage(){ cat << EOU
scontext_test.sh
=================

EOU
}

name=scontext_test
bin=/tmp/$name

defarg="build_run"
arg=${1:-$defarg}

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

CUDA_LIBDIR=$CUDA_PREFIX/lib
[ ! -d "$CUDA_LIBDIR" ] && CUDA_LIBDIR=$CUDA_PREFIX/lib64


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc  \
       -std=c++11 -lstdc++ \
       -I.. \
       -I$CUDA_PREFIX/include \
       -L$CUDA_LIBDIR \
       -lcudart \
       -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

exit 0 


