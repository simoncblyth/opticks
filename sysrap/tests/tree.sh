#!/bin/bash -l

name=TreePruneTest 

gcc $name.cc -g -I. -std=c++11 -lstdc++ -o /tmp/$name 
[ $? -ne 0 ] && echo compile fail && exit 1


#export VERBOSE=1

lldb__ /tmp/$name
#/tmp/$name
[ $? -ne 0 ] && echo run fail && exit 2

exit 0 
 
