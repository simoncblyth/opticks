#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/SPMTAccessor_test.sh

EOU
}

name=SPMTAccessor_test

cd $(dirname $(realpath $BASH_SOURCE))
source $HOME/.opticks/GEOM/GEOM.sh

defarg="info_build_run_ana"
arg=${1:-$defarg}

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export FOLD=$TMP/$name
mkdir -p $FOLD
mkdir -p ${FOLD}.build

bin=${FOLD}.build/$name
script=$name.py

export SPMTAccessor__VERBOSE=1

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="arg name PWD GEOM TMP FOLD CUDA_PREFIX bin"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$HOME/customgeant4 \
           -DWITH_CUSTOM4 \
           -g -std=c++17 -lstdc++ \
           -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
    echo $BASH_SOURCE : build OK
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 4
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 5
fi


exit 0

