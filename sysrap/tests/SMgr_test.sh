#!/usr/bin/env bash
usage(){ cat << EOU

~/o/sysrap/tests/SMgr_test.sh

EOU
}

name=SMgr_test

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

vv="BASH_SOURCE name FOLD"
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done

gcc $name.cc -std=c++17 -lstdc++ -lm -I.. -I$CUDA_PREFIX/include -I$OPTICKS_PREFIX/externals/plog/include -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0



