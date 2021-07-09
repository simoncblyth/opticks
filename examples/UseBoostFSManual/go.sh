#!/bin/bash -l 


name=UseBoostFSManual

BOOST_PREFIX=/usr/local/opticks_externals/boost
mkdir -p /tmp/$name

gcc $name.cc \
     -std=c++11 \
      -I$BOOST_PREFIX/include \
      -L$BOOST_PREFIX/lib \
        -lstdc++ \
       -lboost_system \
       -lboost_filesystem \
       -o /tmp/$name/$name 

[ $? -ne 0 ] && echo compile error && exit 1


/tmp/$name/$name $* 
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

