#!/bin/bash 
usage(){ cat << EOU

~/o/sysrap/tests/srngcpu_test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=srngcpu_test 
bin=/tmp/$name

gcc $name.cc -I.. -std=c++11 -lstdc++ -lm -o $bin && $bin

