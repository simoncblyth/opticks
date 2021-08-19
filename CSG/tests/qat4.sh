#!/bin/bash -l 

glm- 
name=qat4 

gcc $name.cc \
     -I.. \
     -I/usr/local/cuda/include \
     -I$(glm-prefix) \
     -std=c++11 \
      -lstdc++ -o /tmp/$name  

[ $? -ne 0 ] && echo compile fail && exit 1

cmd="/tmp/$name"
echo $cmd
eval $cmd
[ $? -ne 0 ] && echo run fail && exit 2

exit 0 


