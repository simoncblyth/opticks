#!/bin/bash
usage(){ cat << EOU
sseq_test.sh
==============

::

    ~/opticks/sysrap/tests/sseq_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sseq_test

source $HOME/.opticks/GEOM/GEOM.sh

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

test=truncation
export TEST=${TEST:-$test}

#executable=G4CXTest
executable=CSGOptiXSMTest
export EXECUTABLE=${EXECUTABLE:-$executable}

version=4
export VERSION=${VERSION:-$version}

evt=A000
export EVT=${EVT:-$evt}


defarg="info_build_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name defarg arg test TEST"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc -std=c++17 -g -lstdc++ -I.. -I$CUDA_PREFIX/include -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi


