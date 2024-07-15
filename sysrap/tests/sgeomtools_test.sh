#!/bin/bash -l 

usage(){ cat << EOU

::
 
   ~/o/sysrap/tests/sgeomtools_test.sh 


EOU
}


name=sgeomtools_test

defarg="info_build_run"
arg=${1:-$defarg}


vars="BASH_SOURCE name defarg arg CUDA_PREFIX FOLD bin"

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

cd $(dirname $(realpath $BASH_SOURCE)) 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "${var}" "${!var}" ; done
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
          -std=c++11 -lstdc++ -lm \
          -I.. \
          -I$CUDA_PREFIX/include \
          -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

exit 0 

