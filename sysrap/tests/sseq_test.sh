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

#executable=G4CXTest 
executable=CSGOptiXSMTest
export EXECUTABLE=${EXECUTABLE:-$executable}

version=4
export VERSION=${VERSION:-$version}

evt=A000
export EVT=${EVT:-$evt}


defarg="build_run"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $SDIR/$name.cc -std=c++11 -lstdc++ -I$SDIR/.. -I$CUDA_PREFIX/include -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 


