#!/bin/bash -l 
usage(){ cat << EOU
QProp_test.sh 
===============

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=QProp_test
source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_build_run_ana"
arg=${1:-$defarg}

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

vars="BASH_SOURCE SDIR FOLD name GEOM bin"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       -g -std=c++11 -lstdc++ \
       -DMOCK_CURAND \
       -I.. \
       -I$OPTICKS_PREFIX/include/SysRap  \
       -I$CUDA_PREFIX/include \
       -I$OPTICKS_PREFIX/externals/glm/glm \
       -I$OPTICKS_PREFIX/externals/plog/include \
       -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $SDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0 

