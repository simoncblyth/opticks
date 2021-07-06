#!/bin/bash -l 

name=qgs 
gcc $name.cc \
      -std=c++11 \
       -I/usr/local/cuda/include \
       -I/usr/local/opticks/include/SysRap \
         -lstdc++ -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1


/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

