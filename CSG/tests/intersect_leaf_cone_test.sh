#!/bin/bash -l 
usage(){ cat << EOU
intersect_leaf_cone_test.sh
=============================

EOU
}

name=intersect_leaf_cone_test
CUDA_PREFIX=/usr/local/cuda  

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc -g \
    $name.cc \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -I${OPTICKS_PREFIX}/include/SysRap \
    -std=c++11  -lstdc++ \
    -o /tmp/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE run  error && exit 2
fi


exit 0 


