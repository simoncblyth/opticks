#!/bin/bash
usage(){ cat << EOU
~/o/sysrap/tests/sqat4Test.sh
EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=sqat4Test

OPTICKS_PREFIX=${OPTICKS_PREFIX:-/usr/local/opticks}
CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}

bin=/tmp/$name

gcc $name.cc \
     -I.. \
     -I${CUDA_PREFIX}/include \
     -I${OPTICKS_PREFIX}/externals/glm/glm \
     -I${OPTICKS_PREFIX}/include/SysRap \
     -std=c++17 \
     -lm \
     -lstdc++ \
     -o $bin

[ $? -ne 0 ] && echo compile fail && exit 1

$bin
[ $? -ne 0 ] && echo run fail && exit 2

exit 0


