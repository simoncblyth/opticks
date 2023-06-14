#!/bin/bash -l 

name=SName_test 
bin=/tmp/$name

gcc $name.cc -std=c++11 -lstdc++ -I.. -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

GEOM=FewPMT $bin
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0
