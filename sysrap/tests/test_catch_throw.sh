#!/bin/bash

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

name=test_catch_throw
export FOLD=$TMP/$name
mkdir -p $FOLD

bin=$FOLD/$name

gcc $name.cc -std=c++17 -lstdc++ -g -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1

#gdb -ex "catch throw" -ex r --args $bin

source dbg__.sh

type dbg__

dbg__ $bin

[ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2

exit 0




