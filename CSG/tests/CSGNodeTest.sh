#!/bin/bash -l 

#source ../env.sh 

name=CSGNodeTest ; 
srcs="$name.cc ../CSGNode.cc"

CUDA_PREFIX=${CUDA_PREFIX:-/usr/local/cuda}


gcc -g \
   $srcs \
   -I.. \
   -lstdc++ -std=c++11 \
   -I${OPTICKS_PREFIX}/externals/glm/glm \
   -I${OPTICKS_PREFIX}/externals/plog/include \
   -I${OPTICKS_PREFIX}/include/SysRap \
   -I${CUDA_PREFIX}/include \
   -L${OPTICKS_PREFIX}/lib \
   -lSysRap \
   -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

