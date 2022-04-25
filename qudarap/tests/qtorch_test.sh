#!/bin/bash -l 
usage(){ cat << EOU
qtorch_test.sh
================

CPU test of CUDA code to generate torch photons using s_mock_curand.h::

   ./qtorch_test.sh build
   ./qtorch_test.sh run
   ./qtorch_test.sh ana
   ./qtorch_test.sh build_run_ana   # default 

EOU
}

name=qtorch_test 
fold=/tmp/$name
mkdir -p $fold

arg=${1:-build_run_ana}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ \
           -DMOCK_CURAND \
           -I.. \
           -I$OPTICKS_PREFIX/include/SysRap  \
           -I/usr/local/cuda/include \
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


