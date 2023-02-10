#!/bin/bash -l 

name=CSGNode_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 

    gcc $name.cc \
         -g \
         -I.. \
         -I/usr/local/cuda/include \
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

