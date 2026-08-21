#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/scode_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

name=scode_test
FOLD=$TMP/$name
mkdir -p $FOLD
mkdir -p $FOLD/example

bin=$FOLD/$name


#gcc $name.cc -std=c++17 -lstdc++ -I.. -o $bin
g++ $name.cc -std=c++17            -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE - gcc error && exit 1


cd $FOLD

cat << EOI > red.glsl
// [red
#include "green.glsl"
// ]red
EOI

cat << EOI > green.glsl
// [green
#include "blue.glsl"
// ]green
EOI

cat << EOI > blue.glsl
// [blue
// ]blue
EOI



cat << EOT > example/top.glsl
// [top
#include "red.glsl"
// ]top
EOT


$bin
[ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2

exit 0

