#!/bin/bash -l 
usage(){ cat << EOU
sppm_test.sh
=============

~/o/sysrap/tests/sppm_test.sh
~/o/sysrap/tests/sppm_test.cc

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
name=sppm_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

export PPM_PATH=$FOLD/$name.ppm

gcc $name.cc \
    -std=c++11 -lstdc++ \
    -I.. \
    -o $bin

[ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 


$bin
[ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2 


open $PPM_PATH
[ $? -ne 0 ] && echo $BASH_SOURCE : opem error && exit 3


exit 0 


