#!/bin/bash 

sdir=$(pwd)
name=UseNVRTC2
bdir=/tmp/$USER/opticks/$name/build 
rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

export CUDA_PREFIX=/usr/local/cuda

clang++ \
      $sdir/$name.cc \
      $sdir/Prog.cc \
      -o $name \
      -std=c++11 \
      -I$CUDA_PREFIX/include \
      -L$CUDA_PREFIX/lib \
      -lnvrtc \
      -framework CUDA \
      -Wl,-rpath,$CUDA_PREFIX/lib


./$name



