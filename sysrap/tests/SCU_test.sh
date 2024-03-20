#!/bin/bash -l 
usage(){ cat << EOU
SCU_test.sh
============

::

    ~/o/sysrap/tests/SCU_test.sh 


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=SCU_test
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}
for l in lib lib64 ; do [ -d "$CUDA_PREFIX/$l" ] && cuda_l=$l ; done

vars="BASH_SOURCE CUDA_PREFIX cuda_l name"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done

gcc $name.cc \
     -std=c++11 -lstdc++ -g \
     -I$CUDA_PREFIX/include \
     -I.. \
     -L$CUDA_PREFIX/$cuda_l -lcudart \
     -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

exit 0 

