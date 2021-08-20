#!/bin/bash -l 

name=CSGSolidTest 
srcs="$name.cc ../CSGSolid.cc"


gcc -g \
   $srcs \
   -I.. \
   -I${OPTICKS_PREFIX}/include/SysRap \
   -I${OPTICKS_PREFIX}/externals/plog/include \
   -lstdc++ -std=c++11 \
   -I/usr/local/cuda/include \
   -L${OPTICKS_PREFIX}/lib \
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
