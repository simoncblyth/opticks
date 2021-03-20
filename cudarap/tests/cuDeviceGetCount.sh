#!/bin/bash

name=cuDeviceGetCount 
tmp=/tmp/$USER/opticks 
mkdir -p $tmp 
nvcc $name.cc \
     -lstdc++ \
     -I/usr/local/cuda/include \
     -L/usr/local/cuda/lib64 \
     -lcuda \
     -o $tmp/$name 

[ $? -ne 0 ] && echo FAIL && exit 1
$tmp/$name
