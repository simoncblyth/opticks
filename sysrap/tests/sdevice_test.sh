#!/bin/bash -l 

name=sdevice_test
bin=/tmp/$name

defarg="build_run"
arg=${1:-$defarg}

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc  \
       -std=c++11 -lstdc++ \
       -I.. \
       -I$CUDA_PREFIX/include \
       -L$CUDA_PREFIX/lib \
       -lcudart \
       -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

exit 0 



