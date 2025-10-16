#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/squadSelectTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=squadSelectTest 
bin=/tmp/$name

defarg=info_build_run
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name bin defarg arg CUDA_PREFIX"

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   nvcc $name.cu -I.. -I$CUDA_PREFIX/include -o $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE - build error && exit 1
fi  

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2
fi

exit 0


