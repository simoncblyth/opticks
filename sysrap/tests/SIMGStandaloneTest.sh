#!/bin/bash -l 
usage(){ cat << EOU
SIMGStandaloneTest.sh
=======================

1. Loads a 4 channel image (many PNG are 4 channel) using SIMG.h 
2. rotates the image using CUDA texture
3. saved rotates image using SIMG.h 

::

    ~/o/sysrap/tests/SIMGStandaloneTest.sh /tmp/CaTS.png /tmp/CaTS_rotated.png  

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

pwd

name=SIMGStandaloneTest 
bin=/tmp/$name

# attempt to suppress "was set but never used" warnings
# from complilation of stb_image.h using the below causing error 
# -Xcudafe "–-diag_suppress=set_but_not_used" 

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

nvcc $name.cu \
     -lstdc++ \
     -std=c++11 \
     -I$OPTICKS_PREFIX/include/SysRap \
     -I$CUDA_PREFIX/include \
     -L$CUDA_PREFIX/lib \
     -lcudart \
     -o $bin
[ $? -ne 0 ] && echo $BASH_SOURCE " compile FAIL && exit 1

$bin $*
[ $? -ne 0 ] && echo $BASH_SOURCE : run FAIL && exit 2

exit 0 

