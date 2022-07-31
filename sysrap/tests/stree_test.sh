#!/bin/bash -l 

defarg="build_run_ana"
arg=${1:-$defarg}


name=stree_test 

export FOLD=/tmp/$USER/opticks/U4TreeTest


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i stree_test.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 



exit 0 


