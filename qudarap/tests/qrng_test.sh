#!/bin/bash 
usage(){ cat << EOU

~/o/qudarap/tests/qrng_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=qrng_test

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE name defarg arg bin"

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc -g -std=c++11 -lstdc++ -I$CUDA_PREFIX/include -I.. -I../../sysrap -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 
if [ "${arg/build}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

exit 0 


