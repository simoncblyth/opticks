#!/bin/bash -l 


name=G4OpBoundaryProcess_GetFacetNormal_Test

export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name


clhep-
g4-


defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
           -std=c++11 -lstdc++ \
           -I$HOME/np \
           -I$(clhep-prefix)/include \
           -I$(g4-prefix)/include/Geant4  \
           -L$(clhep-prefix)/lib \
           -lCLHEP \
           -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0



