#!/bin/bash -l

msg="=== $BASH_SOURCE :"
name=CRngTest


nvcc ../CRng.cu -c  -o /tmp/CRng.o 
[ $? -ne 0 ] && echo $msg nvcc compile FAIL && exit 1 

gcc \
    $name.cc \
    ../CRng.cc \
    ../CUt.cc \
     /tmp/CRng.o \
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


CRng=INFO /tmp/$name 
[ $? -ne 0 ] && echo $msg run FAIL && exit 1 

exit 0 


