#!/bin/bash -l 
usage(){ cat << EOU
stree_create_test.sh 
======================

::

   ~/o/sysrap/tests/stree_create_test.sh  

build
    standalone compile 
run
    create geometry from scratch and compares query results with expectations
    then persists the geometry to file
ana 
    ipython loads the geometry for examination 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

defarg="build_run_ana"
arg=${1:-$defarg}

name=stree_create_test 

export FOLD=/tmp/$name
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p $FOLD
    gcc $name.cc \
          ../sn.cc \
          ../s_bb.cc \
          ../s_pa.cc \
          ../s_tv.cc \
          ../s_csg.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -DWITH_CHILD \
          -I$CUDA_PREFIX/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 

