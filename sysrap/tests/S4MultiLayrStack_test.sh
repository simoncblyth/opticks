#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/S4MultiLayrStack_test.sh

EOU
}

name=S4MultiLayrStack_test
cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD
bin=$FOLD/$name

opt_scomplex0=-DWITH_SCOMPLEX
opt_scomplex1=-DWITH_SCOMPLEX
opt_thrust=-DWITH_THRUST
opt_std=-DDUMMY

bin_scomplex0=${bin}_scomplex0
bin_scomplex1=${bin}_scomplex1
bin_thrust=${bin}_thrust
bin_std=${bin}_std

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

defarg="info_scomplex0_scomplex1_thrust_std"
arg=${1:-$defarg}

vv="BASH_SOURCE name PWD tmp TMP FOLD bin defarg arg cuda_prefix CUDA_PREFIX" 
if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ scomplex0 ]]; then
    gcc $name.cc -std=c++17 -lstdc++ -lm   -I.. $opt_scomplex0 -o $bin_scomplex0
    [ $? -ne 0 ] && echo $BASH_SOURCE - scomplex0 FAIL && exit 1

    $bin_scomplex0
    [ $? -ne 0 ] && echo $BASH_SOURCE - bin_scomplex0 FAIL && exit 2
fi

if [[ "$arg" =~ scomplex1 ]]; then
    nvcc $name.cc -std=c++17 -lstdc++ -lm   -I.. $opt_scomplex1 -o $bin_scomplex1
    [ $? -ne 0 ] && echo $BASH_SOURCE - scomplex1 FAIL && exit 1

    $bin_scomplex1
    [ $? -ne 0 ] && echo $BASH_SOURCE - bin_scomplex1 FAIL && exit 2
fi


if [[ "$arg" =~ std ]]; then
    gcc $name.cc -std=c++17 -lstdc++ -lm   -I.. $opt_std -o $bin_std
    [ $? -ne 0 ] && echo $BASH_SOURCE - gcc_std FAIL && exit 1

    $bin_std
    [ $? -ne 0 ] && echo $BASH_SOURCE - bin_std FAIL && exit 2
fi


if [[ "$arg" =~ thrust ]]; then
    nvcc $name.cc -std=c++17 -lstdc++ -lm -I${CUDA_PREFIX}/include/cccl  -I.. $opt_thrust -o $bin_thrust
    [ $? -ne 0 ] && echo $BASH_SOURCE - gcc_thrust FAIL && exit 1

    $bin_thrust
    [ $? -ne 0 ] && echo $BASH_SOURCE - bin_thrust FAIL && exit 2
fi

exit 0

