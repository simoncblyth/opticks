#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/scuda_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
name=scuda_test 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg=info_build_run
arg=${1:-$defarg}

#test=ALL
test=uint4_skipahead,uint4_increment
export TEST=${TEST:-$test}

vars="BASH_SOURCE name FOLD CUDA_PREFIX bin defarg arg test TEST"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc  -std=c++11 -lstdc++ -lm  -I.. -I${CUDA_PREFIX}/include -o $bin 
   [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

exit 0 

