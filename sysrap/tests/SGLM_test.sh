#!/bin/bash  
usage(){ cat << EOU
SGLM_test.sh
============

::

    ~/o/sysrap/tests/SGLM_test.sh 

EOU
}

msg="=== $BASH_SOURCE :"
name=SGLM_test 
bin=/tmp/$name

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

cd $(dirname $(realpath $BASH_SOURCE))

#test=Dump
test=descProjection
export TEST=${TEST:-$test}

export VIZMASK=t1

cam=perspective
#cam=orthographic
export CAM=${CAM:-$cam}


vars="BASH_SOURCE name bin CUDA_PREFIX VIZMASK test TEST CAM"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done  

gcc $name.cc -g -Wall -std=c++11 -lstdc++ -lm -I.. -I$OPTICKS_PREFIX/externals/glm/glm -I$CUDA_PREFIX/include -o $bin
[ $? -ne 0 ] && echo $msg compile error && exit 1 

$bin
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 


