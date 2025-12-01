#!/bin/bash

cd $(dirname $(realpath $BASH_SOURCE))

name=sevent_test
bin=/tmp/$name

gcc $name.cc -std=c++17 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/plog/include -lm -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE gcc error && exit 1

$bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0

