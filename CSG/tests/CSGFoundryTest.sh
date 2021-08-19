#!/bin/bash -l

source ../env.sh 

CUDA_PREFIX=/usr/local/cuda   # just use some CUDA headers, not using GPU 

#opts="-DDEBUG=1"
opts=""

name=CSGFoundryTest
srcs="$name.cc ../CSGFoundry.cc ../CSGSolid.cc ../CSGPrim.cc ../CSGPrimSpec.cc ../CSGNode.cc ../CU.cc ../Tran.cc ../Util.cc"
echo compiling $srcs
gcc -g \
       $srcs \
       -std=c++11 \
       -I.. \
       -I${CUDA_PREFIX}/include \
       -I$PREFIX/externals/glm/glm \
       -L${CUDA_PREFIX}/lib -lcudart \
       -lstdc++ $opts \
       -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1
echo compile done 

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH dbg=lldb_  ;;
  Linux)  var=LD_LIBRARY_PATH   dbg=gdb    ;;
esac
#dbg=""
echo var $var dbg $dbg

mkdir -p /tmp/CSGFoundryTest_

cmd="$var=${CUDA_PREFIX}/lib $dbg /tmp/$name $*"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run error && exit 2


exit 0 

