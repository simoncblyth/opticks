#!/bin/bash -l 
usage(){ cat << EOU
SUniformRand_test.sh
======================


EOU
}

name=SUniformRand_test 

export FOLD=/tmp/$name
mkdir -p $FOLD

defarg="build_run_ana"
arg=${1:-$defarg}


clhep-
g4-


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
       -std=c++11 -lstdc++ \
       -I$(clhep-prefix)/include \
       -I$(g4-prefix)/include/Geant4  \
       -L$(clhep-prefix)/lib \
       -lCLHEP \
       -I.. \
       -o /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 3
fi 

exit 0 



