#!/bin/bash -l
msg="=== $BASH_SOURCE :"
name=SCanvasTest 

gcc $name.cc -std=c++11 -I.. -lstdc++ -Wsign-compare -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1

VERBOSE=1 /tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 

