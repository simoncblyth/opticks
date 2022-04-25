#!/bin/bash -l 
usage(){ cat << EOU
storch_test.sh
================

CPU test of CUDA code to generate torch photons using s_mock_curand.h::

   ./storch_test.sh build
   ./storch_test.sh run
   ./storch_test.sh ana
   ./storch_test.sh build_run_ana   # default 

EOU
}

name=storch_test 
fold=/tmp/$name
mkdir -p $fold

arg=${1:-build_run_ana}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ \
           -DMOCK_CURAND \
           -I.. \
           -I/usr/local/cuda/include \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -L$OPTICKS_PREFIX/lib \
           -lSysRap \
           -o $fold/$name 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $fold/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=$fold
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi

exit 0 


