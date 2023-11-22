#!/bin/bash -l 
usage(){ cat << EOU
sseq_test.sh
==============

::

    ~/opticks/sysrap/tests/sseq_test.sh

EOU
}

name=sseq_test 

source $HOME/.opticks/GEOM/GEOM.sh 

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}


gcc $SDIR/$name.cc \
    -std=c++11 -lstdc++ -I$SDIR/.. \
      -I$CUDA_PREFIX/include -o $bin && $bin




