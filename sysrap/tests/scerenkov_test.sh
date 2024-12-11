#!/bin/bash
usage(){ cat << EOU
scerenkov_test.sh
================

CPU test of CUDA code to generate cerenkov photons using srngcpu.h as RNG::

   ~/o/sysrap/tests/scerenkov_test.sh

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
name=scerenkov_test 

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py 

defarg=build_run_ana
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}




if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -lm \
           -DMOCK_CURAND \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -L$OPTICKS_PREFIX/lib64 \
           -lSysRap \
           -o $bin

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/pdb}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $msg pdb error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${PYTHON:-python}  $script
    [ $? -ne 0 ] && echo $msg ana error && exit 4
fi

exit 0 


