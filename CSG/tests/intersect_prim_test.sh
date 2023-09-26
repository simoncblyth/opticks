#!/bin/bash -l 
usage(){ cat << EOU
intersect_prim_test.sh 
=============================

::

    ~/opticks/CSG/tests/intersect_prim_test.sh

EOU
}

cd $(dirname $BASH_SOURCE)
name=intersect_prim_test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="info_build_run"
arg=${1:-$defarg}

vars="BASH_SOURCE name FOLD bin CUDA_PREFIX arg script"

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       ../CSGNode.cc \
       -I..  \
       -std=c++11 -lstdc++ \
       -I${OPTICKS_PREFIX}/externals/plog/include \
       -I${OPTICKS_PREFIX}/include/OKConf \
       -I${OPTICKS_PREFIX}/include/SysRap \
       -L${OPTICKS_PREFIX}/lib \
       -lOKConf -lSysRap \
       -I${CUDA_PREFIX}/include \
       -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

exit 0 

