#!/bin/bash
usage(){ cat << EOU
stmm_vs_sboundary_test.sh
==========================

::

   ~/o/sysrap/tests/stmm_vs_sboundary_test.sh


EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=stmm_vs_sboundary_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py

defarg=build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc \
          -std=c++11 -lstdc++ -lm -g \
          -I.. \
          -DMOCK_CURAND \
          -I$CUDA_PREFIX/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -I$OPTICKS_PREFIX/externals/plog/include \
          -o $bin

   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/pdb}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${PYTHON:-python} $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4 
fi 

exit 0 

