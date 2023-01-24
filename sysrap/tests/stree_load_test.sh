#!/bin/bash -l 
usage(){ cat << EOU
stree_load_test.sh 
=====================

EOU
}

defarg="build_run_ana"
arg=${1:-$defarg}

name=stree_load_test 

export BASE=/tmp/$USER/opticks/U4TreeCreateTest 
export FOLD=$BASE/stree

if [ ! -d "$BASE/stree" ]; then
    echo $BASH_SOURCE : BASE $BASE
    echo $BASH_SOURCE : BASE directory MUST contain an stree directory : THIS DOES NOT 
    exit 1
fi 

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p /tmp/$name
    gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -I/usr/local/cuda/include \
          -I$OPTICKS_PREFIX/externals/glm/glm \
          -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 

