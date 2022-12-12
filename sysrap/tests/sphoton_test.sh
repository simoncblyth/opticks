#!/bin/bash -l 

name=sphoton_test 

defarg="build_run"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ \
           -I.. \
           -I/usr/local/cuda/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -o $FOLD/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $FOLD/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

exit 0 


