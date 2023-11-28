#!/bin/bash -l 
usage(){ cat << EOU
SEvt_test.sh
=============

::

   ~/opticks/sysrap/tests/SEvt_test.sh 


EOU
}

name=SEvt_test 
cd $(dirname $BASH_SOURCE) 

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name 

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

gcc $name.cc \
    -std=c++11 -lstdc++ \
    -I.. \
    -I$(opticks-prefix)/externals/plog/include \
    -I$(opticks-prefix)/externals/glm/glm \
    -I$CUDA_PREFIX/include \
    -L$(opticks-prefix)/lib \
    -lSysRap \
    -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE : compile error && exit 1 

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 

exit 0 

