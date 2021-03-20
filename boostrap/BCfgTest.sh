#!/bin/bash
name=BCfgTest 
tmp=/tmp/$USER/opticks 
mkdir -p $tmp

gcc $name.cc -lstdc++ -std=c++11 -I.. -I$OPTICKS_PREFIX/externals/plog/include -o $tmp/$name 
[ ! $? -eq 0 ] && echo fail && exit 1

$tmp/$name

