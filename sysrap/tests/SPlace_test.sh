#!/bin/bash -l 

name=SPlace_test 

export FOLD=/tmp/$name 
export TEST=${TEST:-AroundCylinder} 

mkdir -p $FOLD 

gcc $name.cc \
      -std=c++11 -lstdc++ \
        -I/usr/local/cuda/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -I.. \
        -o $FOLD/$name 

[ $? -ne 0 ] && echo == $BASH_SOURCE compile error && exit 1 

$FOLD/$name
[ $? -ne 0 ] && echo == $BASH_SOURCE run error && exit 2 


${IPYTHON:-ipython} --pdb -i $name.py
[ $? -ne 0 ] && echo == $BASH_SOURCE ana error && exit 3

exit 0 


