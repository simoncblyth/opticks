#!/bin/bash -l 

name=sphoton_test 

mkdir -p /tmp/$name 
gcc $name.cc -std=c++11 -lstdc++ -I.. -I/usr/local/cuda/include -o /tmp/$name/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

/tmp/$name/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2


exit 0 


