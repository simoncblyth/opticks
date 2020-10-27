#!/bin/bash -l

# /Developer/NVIDIA/CUDA-9.1/doc/pdf/NVRTC_User_Guide.pdf


sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/opticks/$name/build 
rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

export CUDA_PREFIX=/usr/local/cuda

clang++ \
      $sdir/$name.cc \
      -o $name \
      -std=c++11 \
      -I$CUDA_PREFIX/include \
      -L$CUDA_PREFIX/lib \
      -lnvrtc \
      -framework CUDA \
      -Wl,-rpath,$CUDA_PREFIX/lib


./$name



