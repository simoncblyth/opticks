#!/bin/bash -l 

name=NPFold_test 
mkdir -p /tmp/$name 

gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name/$name
[ $? -ne 0 ] && echo $msg run error && exit 2

exit 0 



