#!/bin/bash -l 
usage(){ cat << EOU
SScene_test.sh
===============

::

   ~/o/sysrap/tests/SScene_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SScene_test
bin=/tmp/$name

stree_fold=$TMP/U4TreeCreateTest
export STREE_FOLD=${STREE_FOLD:-$stree_fold}

if [ ! -d "$STREE_FOLD/stree" ]; then
   echo $BASH_SOURCE : ERROR STREE_FOLD $STREE_FOLD DOES NOT CONTAIN stree 
   exit 1
fi 

vars="BASH_SOURCE PWD stree_fold STREE_FOLD bin"

defarg=info_build_run
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

glm-

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi   

if [ "${arg/build}" != "$arg" ]; then 

    gcc  $name.cc \
         ../sn.cc \
         ../s_pa.cc \
         ../s_tv.cc \
         ../s_bb.cc \
         ../s_csg.cc \
         -DWITH_CHILD \
         -std=c++11 -lstdc++ \
         -I.. \
         -I$(glm-prefix) \
         -I${CUDA_PREFIX}/include \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 


exit 0 
