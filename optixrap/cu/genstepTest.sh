#!/bin/bash -l 

name=genstepTest

gcc $name.cc \
      -std=c++11 -lstdc++ \
      -I$HOME/np \
      -I/usr/local/cuda/include \
      -I/usr/local/opticks/include/OpticksCore \
      -o /tmp/$name 

[ $? -ne 0 ] && echo $msg compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $msg run error && exit 1 

exit 0 
