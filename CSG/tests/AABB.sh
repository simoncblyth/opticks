#!/bin/bash -l 

name=AABB 

gcc $name.cc \
     -I.. \
     -I/usr/local/cuda/include \
     -std=c++11 \
     -lstdc++ -o /tmp/$name  

[ $? -ne 0 ] && echo compile fail && exit 1

cmd="/tmp/$name"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run fail && exit 2

exit 0 


