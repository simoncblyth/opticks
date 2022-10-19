#!/bin/bash -l 
usage(){ cat << EOU
OKConf_test.sh 
================

Testing the header-only OKConf.h 

EOU
}

name=OKConf_test 

gcc $name.cc \
      -std=c++11 -lstdc++ \
      -I.. \
      -I$OPTICKS_PREFIX/include/OKConf \
       -o /tmp/$name 

[ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2

exit 0 


