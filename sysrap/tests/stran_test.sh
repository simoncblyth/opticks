#!/bin/bash -l

name=stran_test 

gcc $name.cc -std=c++11 -lstdc++ -I/usr/local/cuda/include -I$OPTICKS_PREFIX/externals/glm/glm -I.. -o /tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE compilation error && exit 1 

/tmp/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2


exit 0 



