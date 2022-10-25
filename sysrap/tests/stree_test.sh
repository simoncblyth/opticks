#!/bin/bash -l 
usage(){ cat << EOU
stree_test.sh 
===============

Tool to load a persisted stree geometry for debugging. 


EOU
}

defarg="build_run_ana"
arg=${1:-$defarg}

name=stree_test 

export GEOM=${GEOM:-J004}
export CFBASE=$HOME/.opticks/GEOM/$GEOM
export BASE=$CFBASE/CSGFoundry/SSim
export FOLD=$BASE/stree

if [ ! -d "$BASE/stree" ]; then
    echo $BASH_SOURCE : BASE $BASE
    echo $BASH_SOURCE : BASE directory MUST contain an stree directory : THIS DOES NOT 
    exit 1
fi 

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE GEOM BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p /tmp/$name
    gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
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

