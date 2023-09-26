#!/bin/bash -l 
usage(){ cat << EOU
U4Mesh_test.sh
================

::

    ~/opticks/u4/tests/U4Mesh_test.sh

EOU
}

cd $(dirname $BASH_SOURCE)
name=U4Mesh_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

clhep-
g4-


vars="BASH_SOURCE name FOLD bin"

defarg="info_build_run_ana"
arg=${1:-$defarg}


[ "${arg/info}" != "$arg" ] && for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc \
         -I.. \
         -std=c++11 -lstdc++ \
         -I$HOME/opticks/sysrap \
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

