#!/bin/bash -l

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

name=CSGPrimTest
srcs="$name.cc ../CSGPrim.cc ../CSGPrimSpec.cc ../CU.cc"

gcc -g \
    $srcs \
    -I.. \
    -I${CUDA_PREFIX}/include \
    -std=c++11  -lstdc++ \
    -L${CUDA_PREFIX}/lib -lcudart  \
    -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH debugger=lldb_  ;;
  Linux)  var=LD_LIBRARY_PATH   debugger=gdb    ;;
esac

cmd="$var=${CUDA_PREFIX}/lib  /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

