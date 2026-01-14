#!/bin/sh
usage(){ cat << EOU

~/o/sysrap/tests/SNameTest.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SNameTest
source dbg__.sh
dbg__ $name

