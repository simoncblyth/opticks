#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

name=QStateTest 
mkdir -p /tmp/$name

gcc $name.cc -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/include/SysRap -I/usr/local/cuda/include -o /tmp/$name/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 


