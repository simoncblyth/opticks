#!/bin/bash -l 
usage(){ cat << EOU

~/o/sysrap/tests/SCU_BufferView_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SCU_BufferView_test

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

gcc $name.cc \
     -g -std=c++11 -lstdc++ \
     -I.. \
     -I$CUDA_PREFIX/include \
     -L$CUDA_PREFIX/$cuda_l -lcudart \
     -o $bin 
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

#dbg__ $bin
$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 


