#!/bin/bash 
usage(){ cat << EOU

~/o/sysrap/tests/scurand_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=scurand_test 
FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name

gcc $name.cc -g -I.. -DMOCK_CURAND -std=c++11 -lstdc++ -lm -o $bin && $bin
