#!/bin/bash -l 

defarg="build_run"
arg=${1:-$defarg}

export BASE=/tmp/$USER/opticks/ntds3/G4CXOpticks

name=stree_sensor_test 
mkdir -p /tmp/$name

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

