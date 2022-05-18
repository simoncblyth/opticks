#!/bin/bash -l 
usage(){ cat << EOU
scarrier_test.sh
================

CPU test of CUDA code to generate carrier photons using s_mock_curand.h::

   ./scarrier_test.sh build
   ./scarrier_test.sh run
   ./scarrier_test.sh ana
   ./scarrier_test.sh build_run_ana   # default 

EOU
}

msg="=== $BASH_SOURCE :"
name=scarrier_test 
fold=/tmp/$USER/opticks/$name
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
    echo $msg FOLD $FOLD
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi

exit 0 


