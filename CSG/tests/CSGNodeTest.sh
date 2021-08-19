#!/bin/bash -l 
source ../env.sh 

name=CSGNodeTest ; 
srcs="$name.cc ../CSGNode.cc"


gcc -g \
   $srcs \
   -I.. \
   -lstdc++ -std=c++11 \
   -I$PREFIX/externals/glm/glm \
   -I/usr/local/cuda/include \
   -o /tmp/$name 

[ $? -ne 0 ] && echo compile error && exit 1 

/tmp/$name
[ $? -ne 0 ] && echo run error && exit 2

exit 0 

