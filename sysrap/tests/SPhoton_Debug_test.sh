#!/bin/bash -l 
usage(){ cat << EOU
SPhoton_Debug_test.sh
======================


EOU
}

name=SPhoton_Debug_test 

export FOLD=/tmp/$name
mkdir -p $FOLD

defarg="build_run_ana"
arg=${1:-$defarg}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
         -std=c++11 -lstdc++ \
         -I${OPTICKS_PREFIX}_externals/g4_1042/include/Geant4 \
         -I${OPTICKS_PREFIX}_externals/clhep_2440/include \
         -I.. -o /tmp/$name/$name

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



