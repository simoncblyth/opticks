#!/bin/bash -l 

defarg="build_run_ana"
arg=${1:-$defarg}


name=stree_test 

#export BASE=/tmp/$USER/opticks/U4TreeTest
export BASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export BASE_aug5=/tmp/$USER/opticks/ntds3_aug5/G4CXOpticks

## gets loaded from STBASE/stree

export FOLD=$BASE/stree
export CFBASE=$BASE
#export CFBASE=$BASE_aug5

#source $OPTICKS_HOME/bin/COMMON.sh 
#T_FOLD=$($OPTICKS_HOME/g4cx/gxt.sh fold)
#T_CFBASE=$(upfind_cfbase $T_FOLD)  


if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE GEOM BASE FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "${arg/build}" != "$arg" ]; then 
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

