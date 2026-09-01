#!/bin/bash
usage(){ cat << EOU
SOPTIX_Capability_test.sh
===========================

::

    ~/o/sysrap/tests/SOPTIX_Capability_test.sh
    ~/o/sysrap/tests/SOPTIX_Capability_test.cc


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SOPTIX_Capability_test
tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

optix_prefix=${OPTICKS_OPTIX_PREFIX}
OPTIX_PREFIX=${OPTIX_PREFIX:-$optix_prefix}


vars="BASH_SOURCE CUDA_PREFIX OPTIX_PREFIX tmp TMP FOLD bin"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done

gcc $name.cc \
    -std=c++17 -lstdc++ -lm -ldl  -g \
    -I.. \
    -I$CUDA_PREFIX/include \
    -I$OPTIX_PREFIX/include \
    -I$OPTICKS_PREFIX/externals/glm/glm \
    -L$CUDA_PREFIX/lib64 -lcudart \
    -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

exit 0

