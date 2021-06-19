#!/bin/bash -l

msg="=== $BASH_SOURCE :"
name=QRngTest


nvcc ../QRng.cu -c  -o /tmp/QRng.o 
[ $? -ne 0 ] && echo $msg nvcc compile FAIL && exit 1 

gcc \
    $name.cc \
    ../QRng.cc \
    ../QU.cc \
     /tmp/QRng.o \
             \
     -std=c++11 \
     -I.. \
     -I/usr/local/cuda/include \
     -I${OPTICKS_PREFIX}/include/SysRap \
     -I${OPTICKS_PREFIX}/externals/plog/include \
     -lstdc++ \
     -L/usr/local/cuda/lib \
     -lcudart \
     -L${OPTICKS_PREFIX}/lib \
     -lSysRap \
     -o /tmp/$name

[ $? -ne 0 ] && echo $msg gcc compile FAIL && exit 1 


QRng=INFO /tmp/$name 
[ $? -ne 0 ] && echo $msg run FAIL && exit 1 

exit 0 


