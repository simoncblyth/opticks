#!/bin/bash -l 

name=SPMTAccessor_test

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_build_run_ana"
arg=${1:-$defarg}

TMP=${TMP:-/tmp/$USER/opticks}

export FOLD=$TMP/$name
mkdir -p $FOLD
mkdir -p ${FOLD}.build

bin=${FOLD}.build/$name


export SPMTAccessor__VERBOSE=1



CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

vars="arg name SDIR GEOM TMP FOLD CUDA_PREFIX bin"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $SDIR/$name.cc \
           -I.. \
           -I$CUDA_PREFIX/include \
           -I$HOME/customgeant4 \
           -DWITH_CUSTOM4 \
           -g -std=c++11 -lstdc++ \
           -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
    echo $BASH_SOURCE : build OK 
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
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

