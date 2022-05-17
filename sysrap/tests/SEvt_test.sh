#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=SEvt_test 

gcc $name.cc \
    -std=c++11 -lstdc++ \
    -I.. \
    -I$(opticks-prefix)/externals/plog/include \
    -I/usr/local/cuda/include \
    -L$(opticks-prefix)/lib \
    -lSysRap \
    -o /tmp/$name 

[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 

