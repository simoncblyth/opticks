#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/SBitSet_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SBitSet_test
tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
bin=$TMP/$name/$name
mkdir -p $(dirname $bin)

export TEST=Roundtrip

vv="BASH_SOURCE PWD name tmp TMP bin TEST"
for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done

gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0

