#!/bin/bash
usage(){ cat << EOU

~/opticks/sysrap/tests/ssystime_test.sh

EOU
}

name=ssystime_test
bin=/tmp/$name

cd $(dirname $(realpath $BASH_SOURCE))

gcc $name.cc -I.. -std=c++11 -lstdc++ -o $bin && $bin


