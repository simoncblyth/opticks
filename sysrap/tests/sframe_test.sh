#!/bin/bash -l

name=sframe_test 
export FOLD=/tmp/$name 
mkdir -p $FOLD

gcc $name.cc \
      -std=c++11 -lstdc++ \
        -I.. \
        -I/usr/local/cuda/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -o $FOLD/$name 

[ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1

$FOLD/$name 
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2


${IPYTHON:-ipython} --pdb -i $name.py 
[ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3





exit 0 


