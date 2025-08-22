#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/snode_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=snode_test
bin=/tmp/$name

gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin && $bin



