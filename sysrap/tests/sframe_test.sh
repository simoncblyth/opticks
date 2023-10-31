#!/bin/bash -l

name=sframe_test 
export FOLD=/tmp/$name 
mkdir -p $FOLD

defarg="build_run_ana"
arg=${1:-$defarg}


if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc \
      -std=c++11 -lstdc++ -lm \
        -I.. \
        -I/usr/local/cuda/include \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -o $FOLD/$name 
   [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $FOLD/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi

exit 0 


