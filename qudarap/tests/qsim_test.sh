#!/bin/bash -l 

name=qsim_test 

gcc $name.cc -std=c++11 -lstdc++ \
       -DMOCK_CURAND \
       -I.. \
       -I$OPTICKS_PREFIX/include/SysRap  \
       -I/usr/local/cuda/include \
       -o /tmp/$name 

[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 2 

exit 0 



