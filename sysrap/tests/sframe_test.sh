#!/bin/bash -l
usage(){ cat << EOU
sframe_test.sh
===============

::

    ~/o/sysrap/tests/sframe_test.sh
    ~/o/sysrap/tests/sframe_test.sh build_run 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sframe_test 
export FOLD=${TMP:-/tmp/$USER/opticks}/$name 
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py 

defarg="build_run_ana"
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc \
      -std=c++11 -lstdc++ -lm \
        -I.. \
        -I$CUDA_PREFIX/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

exit 0 


