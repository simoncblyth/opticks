#!/bin/bash -l 

name=CSGClassifyTest 

gcc $name.cc -std=c++11 -I.. -I$OPTICKS_PREFIX/include/sysrap -Wsign-compare -lstdc++ -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name $*


 
