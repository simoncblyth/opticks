#!/bin/bash -l 


name=intersect_leaf_cone_test
CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

gcc -g \
    $name.cc \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -I${OPTICKS_PREFIX}/include/SysRap \
    -std=c++11  -lstdc++ \
    -o /tmp/$name 

[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1

/tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE run  error && exit 2

exit 0 


