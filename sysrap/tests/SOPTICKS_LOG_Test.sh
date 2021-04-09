#!/bin/bash -l 

plog-
name=SOPTICKS_LOG_Test
bin=/tmp/$name

gcc $name.cc \
      -std=c++11 \
      -lstdc++ \
      -I.. \
      -DOPTICKS_SYSRAP \
      -I$(plog-prefix)/include \
      -L$(opticks-prefix)/lib -lSysRap \
       -o $bin

[ $? -ne 0 ] && echo compile error && exit 1

$bin
[ $? -ne 0 ] && echo run error && exit 2



exit 0 
