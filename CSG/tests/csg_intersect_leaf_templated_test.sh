#!/bin/bash -l 

name=csg_intersect_leaf_templated_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


defarg="info_build_run"
arg=${1:-$defarg}

opt="-DWITH_SCUDA_DOUBLE"

vars="BASH_SOURCE name FOLD bin CUDA_PREFIX arg opt"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
      $opt \
      -std=c++11 -lstdc++ \
       -I..  \
       -I../../sysrap \
       -I${CUDA_PREFIX}/include \
        -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

exit 0 

