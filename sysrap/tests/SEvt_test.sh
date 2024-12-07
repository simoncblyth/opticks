#!/bin/bash
usage(){ cat << EOU
SEvt_test.sh
=============

::

   ~/opticks/sysrap/tests/SEvt_test.sh 


EOU
}

name=SEvt_test 
cd $(dirname $(realpath $BASH_SOURCE)) 
source dbg__.sh 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg=info_build_run
arg=${1:-$defarg}

vars="BASH_SOURCE 0 name bin TMP FOLD defarg arg"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi 

if [ "${arg/build}" != "$arg" ]; then
   gcc $name.cc \
      -std=c++11 -lstdc++ \
      -I.. \
      -g \
      -I$OPTICKS_PREFIX/externals/plog/include \
      -I$OPTICKS_PREFIX/externals/glm/glm \
      -I$CUDA_PREFIX/include \
      -L$OPTICKS_PREFIX/lib64 \
      -lSysRap \
      -lm \
      -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3 
fi 

exit 0 

