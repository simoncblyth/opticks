#!/usr/bin/env bash

usage(){ cat << EOU
QTexLayeredLookupTest.sh
=========================

Roundtrip test of a layered 2D texture.

~/o/qudarap/tests/QTexLayeredLookupTest.sh

EOU
}

defarg=info_nvcc_gcc_run_ls_ana
arg=${1:-$defarg}

cd $(dirname $(realpath $BASH_SOURCE))

name=QTexLayeredLookupTest
script=$name.py

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vv="BASH_SOURCE PWD defarg arg name cuda_prefix CUDA_PREFIX bin tmp TMP FOLD script"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/nvcc}" != "$arg" ]; then

    echo $BASH_SOURCE $LINENO nvcc
    nvcc -c ../QTexLayeredLookup.cu  \
         -std=c++17 -lstdc++ \
         -I.. \
         -I../../sysrap \
         -o $FOLD/QTexLayeredLookup_cu.o
    [ $? -ne 0 ] && echo $BASH_SOURCE : nvcc compile error $name.cu  && exit 1
fi


if [ "${arg/gcc}" != "$arg" ]; then
    echo $BASH_SOURCE $LINENO gcc
    gcc $name.cc  \
         -g -std=c++17 -lstdc++ -lcudart \
         -DWITH_CUDA \
         $FOLD/QTexLayeredLookup_cu.o \
         -I.. \
         -I../../sysrap \
         -I$CUDA_PREFIX/include \
         -L$CUDA_PREFIX/lib64 \
         -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : gcc error && exit 2
fi

if [ "${arg/run}" != "$arg" ]; then
    echo $BASH_SOURCE $LINENO run
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - run fail && exit 1
fi

if [ "${arg/ls}" != "$arg" ]; then
    echo ls -alst $FOLD
    ls -alst $FOLD
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ls fail && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - pdb fail && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $script
    [ $? -ne 0 ] && echo $BASH_SOURCE $LINENO - ana fail && exit 4
fi
exit 0

