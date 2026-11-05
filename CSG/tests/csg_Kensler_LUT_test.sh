#!/bin/bash
usage(){ cat << EOU
csg_Kensler_LUT_test.sh
=========================

~/o/CSG/tests/csg_Kensler_LUT_test.sh


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=csg_Kensler_LUT_test

test=LUT
export TEST=${TEST:-$test}

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name
script=$name.py

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="info_build_run_pdb"
arg=${1:-$defarg}

vars="BASH_SOURCE PWD TEST name TMP FOLD bin CUDA_PREFIX arg script"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc \
       $name.cc \
       -std=c++17 -lstdc++ -lm \
       -I..  \
       -I${OPTICKS_PREFIX}/include/SysRap \
       -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1

    #   -L${CUDA_PREFIX}/lib64 -lcudart -I${CUDA_PREFIX}/include
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 4
fi

exit 0


