#!/bin/bash -l 

name=squadTest 

OPTICKS_PREFIX=${OPTICKS_PREFIX:-/usr/local/opticks}
CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

gcc $name.cc \
     -I.. \
     -I${CUDA_PREFIX}/include \
     -I${OPTICKS_PREFIX}/include/SysRap \
     -std=c++11 \
     -lstdc++ \
     -o /tmp/$name  

[ $? -ne 0 ] && echo compile fail && exit 1

source ../../qudarap/tests/ephoton.sh 

cmd="/tmp/$name"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run fail && exit 2

exit 0 


