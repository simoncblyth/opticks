#!/bin/bash 

usage(){ cat << EOU

~/o/sysrap/tests/s_mock_curand_test.sh 


EOU
}
cd $(dirname $(realpath $BASH_SOURCE))

name=s_mock_curand_test  
bin=/tmp/$name
gcc $name.cc -std=c++11 -lstdc++ -lm -g -I.. -o $bin && $bin



