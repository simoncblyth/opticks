#!/bin/bash -l 
usage(){ cat << EOU
QBnd_test.sh 
=============

Via MOCK_TEXTURE and MOCK_CUDA switches this builds and 
runs CUDA code including texture lookups on the CPU via
mocking of the texture handling. 

EOU
}

name=QBnd_test
source $HOME/.opticks/GEOM/GEOM.sh 

defarg="info_build_run_ana"
arg=${1:-$defarg}
export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vars="BASH_SOURCE arg name GEOM FOLD CUDA_PREFIX OPTICKS_PREFIX"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 

    gcc $name.cc ../QBnd.cc ../QTex.cc ../QOptical.cc  \
         -g -std=c++11 -lstdc++ \
         -DMOCK_TEXTURE \
         -DMOCK_CUDA \
         -I.. \
         -I$OPTICKS_PREFIX/include/SysRap \
         -I$CUDA_PREFIX/include \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 
fi 


if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
       Darwin) lldb__ $bin ;; 
       Linux)  gdb__ $bin  ;;
    esac 
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3 
fi 



if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py  
    [ $? -ne 0 ] && echo $BASH_SOURCE : eun error && exit 4 
fi 






exit 0 
