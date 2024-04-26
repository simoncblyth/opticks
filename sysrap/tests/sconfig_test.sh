#!/bin/bash -l 
usage(){ cat << EOU
sconfig_test.sh
===============

~/o/sysrap/tests/sconfig_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sconfig_test
bin=/tmp/$name

gcc $name.cc \
    -std=c++11 -lstdc++ \
    -I$OPTICKS_PREFIX/include/SysRap \
    -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 


$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2

exit 0  

