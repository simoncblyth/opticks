#!/bin/bash -l 
usage(){ cat << EOU
SPMT_test.sh
===============

::

    N_MCT=900 N_SPOL=2  ./SPMT_test.sh  

EOU
}


REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
source $HOME/.opticks/GEOM/GEOM.sh 

name=SPMT_test
defarg="build_run_ana"
arg=${1:-$defarg}

FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

export SFOLD=/tmp/$name
export JFOLD=/tmp/JPMTTest


CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

vars="arg name REALDIR GEOM FOLD SFOLD JFOLD"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $REALDIR/$name.cc \
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
    case $(uname) in 
       Darwin) lldb__ $bin ;;
       Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3 
fi 


if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 


exit 0 


