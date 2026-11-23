#!/bin/bash


cd $(dirname $(realpath $BASH_SOURCE))
name=CSGSolidTest 
script=$name.py



gcc -g \
   $name.cc \
   ../CSGSolid.cc \
   -I.. \
   -I${OPTICKS_PREFIX}/include/SysRap \
   -I${OPTICKS_PREFIX}/externals/plog/include \
   -lstdc++ -std=c++17 \
   -I/usr/local/cuda/include \
   -L${OPTICKS_PREFIX}/lib64 \
   -lSysRap \
   -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1


path=/tmp/$name.npy
/tmp/$name $path
[ $? -ne 0 ] && echo run error && exit 2


ls -l $path

ipython CSGSolidTest.py $path
[ $? -ne 0 ] && echo ana error && exit 3


exit 0
