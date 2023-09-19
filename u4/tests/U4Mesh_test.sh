#!/bin/bash -l 

name=U4Mesh_test

FOLD=/tmp/$name
mkdir -p $FOLD
export FOLD

bin=$FOLD/$name
vars="BASH_SOURCE name FOLD bin"


defarg="info_build_run_ana"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

clhep-
g4-

if [ "${arg/build}" != "$arg" ]; then
    gcc \
         $name.cc \
         -I.. \
         -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
         -I$CUDA_PREFIX/include \
         -I$OPTICKS_PREFIX/externals/glm/glm \
         -I$(clhep-prefix)/include \
         -I$(g4-prefix)/include/Geant4  \
         -L$(g4-prefix)/lib \
         -L$(clhep-prefix)/lib \
         -lG4global \
         -lG4geometry \
         -lG4graphics_reps \
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


