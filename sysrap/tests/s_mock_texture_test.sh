#!/bin/bash -l 
usage(){ cat << EOU
s_mock_texture_test.sh
=======================


EOU
}

name=s_mock_texture_test

defarg="info_build_run_ana"
arg=${1:-$defarg}

source $HOME/.opticks/GEOM/GEOM.sh 

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE name arg GEOM CUDA_PREFIX"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
        -g -std=c++11 -lstdc++ \
        -I.. \
        -I${CUDA_PREFIX}/include  \
        -o $bin 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi







exit 0 
