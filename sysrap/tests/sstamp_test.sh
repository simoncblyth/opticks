#!/bin/bash 

usage(){ cat << EOU

~/o/sysrap/tests/sstamp_test.sh 

EOU
}



name="sstamp_test"

FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

cd $(dirname $(realpath $BASH_SOURCE))

gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

$bin 
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0  



