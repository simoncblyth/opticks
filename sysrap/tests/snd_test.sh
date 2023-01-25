#!/bin/bash -l 

name=snd_test 

gcc $name.cc \
    -std=c++11 -lstdc++ \
    -I.. \
    -I/usr/local/cuda/include \
    -I$OPTICKS_PREFIX/externals/glm/glm \
    -o /tmp/$name 

[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 


