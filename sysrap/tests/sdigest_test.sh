#!/bin/bash -l 

name=sdigest_test 

gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 


