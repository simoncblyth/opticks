#!/bin/bash -l 

name=randomTest 
gcc $name.cc -std=c++11 -lstdc++ -I$HOME/np -I/usr/local/cuda/include -o /tmp/$name 
[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 1 

exit 0 
