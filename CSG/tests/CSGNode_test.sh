#!/bin/bash -l 
usage(){ cat << EOU
CSGNode_test.sh
=================

::

   ~/o/CSG/tests/CSGNode_test.sh

EOU
}

name=CSGNode_test

cd $(dirname $(realpath $BASH_SOURCE))

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 

    gcc $name.cc \
         -g \
         -I.. \
         -I$CUDA_PREFIX/include \
         -I$OPTICKS_PREFIX/externals/plog/include \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$OPTICKS_PREFIX/include/SysRap \
         -std=c++11 -lstdc++ \
         -L$OPTICKS_PREFIX/lib \
         -lSysRap -lCSG \
         -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

exit 0

