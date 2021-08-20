#!/bin/bash -l 

name=AABB 

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

gcc $name.cc \
     -I.. \
     -I${CUDA_PREFIX}/include \
     -std=c++11 \
     -lstdc++ -o /tmp/$name  

[ $? -ne 0 ] && echo compile fail && exit 1

cmd="/tmp/$name"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run fail && exit 2

exit 0 


