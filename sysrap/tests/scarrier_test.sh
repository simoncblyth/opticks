#!/bin/bash
usage(){ cat << EOU
scarrier_test.sh
================

CPU test of CUDA code to generate carrier photons using srngcpu.h::

    ~/o/sysrap/tests/scarrier_test.sh 

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=scarrier_test 
export FOLD=/tmp/$USER/opticks/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py 

defarg=info_build_run_ana
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE FOLD name bin script PWD defarg arg OPTICKS_PREFIX"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -g -lm \
           -DMOCK_CURAND \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -L$OPTICKS_PREFIX/lib64 \
           -lSysRap -lm \
           -o $bin

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi



exit 0 


