#!/bin/bash -l 

defarg="build_run_ana"
arg=${1:-$defarg}


name=stree_test 

#export STBASE=/tmp/$USER/opticks/U4TreeTest
export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export FOLD=$STBASE/stree

source $OPTICKS_HOME/bin/COMMON.sh 
T_FOLD=$($OPTICKS_HOME/g4cx/gxt.sh fold)
T_CFBASE=$(upfind_cfbase $T_FOLD)  

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE GEOM T_FOLD T_CFBASE STBASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

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
    export CFBASE=${CFBASE:-$T_CFBASE}
    ${IPYTHON:-ipython} --pdb -i stree_test.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

exit 0 

