#!/bin/bash -l 
usage(){ cat << EOU
CSG_stree_Convert_test.sh
===========================



EOU
}

source $HOME/.opticks/GEOM/GEOM.sh 

name=CSG_stree_Convert_test  

export FOLD=/tmp/$name
bin=$FOLD/$name
mkdir -p $FOLD

defarg="info_build_run"
arg=${1:-$defarg}

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE GEOM FOLD"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
           -I${CUDA_PREFIX}/include \
           -I$OPTICKS_PREFIX/externals/glm/glm \
           -I$OPTICKS_PREFIX/externals/plog/include \
           -I$OPTICKS_PREFIX/include/SysRap \
           -I$OPTICKS_PREFIX/include/CSG \
           -L$OPTICKS_PREFIX/lib \
           -lSysRap \
           -lCSG \
           -std=c++11 -lstdc++ \
           -o $bin 

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 

exit 0 
