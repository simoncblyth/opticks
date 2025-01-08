#!/bin/bash 
usage(){ cat << EOU 
sbuild_test.sh
==============

~/o/sysrap/tests/sbuild_test.sh 

NB sbuild_test can be built locally by this script
OR by the standard CMake build

Using the standard binary off the PATH::

    TEST=RNGName sbuild_test
    Philox

    TEST=BuildType sbuild_test
    Debug

    TEST=ContextString sbuild_test
    Debug_Philox


EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=sbuild_test
bin=/tmp/$name

opt="-DCONFIG_Release -DDEBUG_TAG -DPRODUCTION -DRNG_XORWOW"
#opt=-DCONFIG_Debug

std=c++17

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


vars="BASH_SOURCE name bin opt std"
for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 

gcc $name.cc -std=$std $opt -lstdc++ -I.. -I$CUDA_PREFIX/include -o $bin 
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 

