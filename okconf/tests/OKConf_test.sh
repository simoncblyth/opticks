#!/bin/bash
usage(){ cat << EOU
OKConf_test.sh
================

~/o/okconf/tests/OKConf_test.sh

Testing the header-only OKConf.h


CAUTION : DESPITE "WITH_CUDA" OKConf.h
DOES NOT USE THE CUDA API - AND HENCE DOES NOT
FORCE LINKING WITH -lcudart

BUILD TIME CUDA VERSION INFO IS DETERMINED ALL AT CMAKE LEVEL
AND COMMUNICATED TO CODE LEVEL VIA CODE GENERATION AND
PREPROCESSOR MACROS

SEE OKConf_CUDART_test.sh FOR RUNTIME DETERMINATION
OF CUDA DRIVER VERSION

EOU
}

name=OKConf_test
bin=/tmp/$name


cd $(dirname $(realpath $BASH_SOURCE))


gcc $name.cc \
      -std=c++17 -lstdc++ \
      -I.. \
      -DWITH_CUDA \
      -I$OPTICKS_PREFIX/include/OKConf \
      -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0


