#!/bin/bash -l 


name=ssys_test 

gcc $name.cc \
       -std=c++11 -lstdc++ \
       -I.. \
       -o /tmp/$name 

[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 

i=-1 u=2 f=101.3 d=-202.5 /tmp/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 

exit 0 

