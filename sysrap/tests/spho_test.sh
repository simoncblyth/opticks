#!/bin/bash -l 

name=spho_test 
export FOLD=/tmp/$name
mkdir -p $FOLD

gcc $name.cc -std=c++11 -lstdc++ -I.. -o $FOLD/$name
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

$FOLD/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 

