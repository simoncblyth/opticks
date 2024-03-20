#!/bin/bash -l 

name=SIMGStandaloneTest 
bin=/tmp/$name

# attempt to suppress "was set but never used" warnings
# from complilation of stb_image.h using the below causing error 
# -Xcudafe "–-diag_suppress=set_but_not_used" 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}


nvcc $name.cu -lstdc++ -std=c++11  -I.. -I. -I$CUDA_PREFIX/include -LCUDA_PREFIX/lib -lcudart -o $bin
[ $? -ne 0 ] && echo compile FAIL && exit 1

$bin $*
[ $? -ne 0 ] && echo run FAIL && exit 2

exit 0 

