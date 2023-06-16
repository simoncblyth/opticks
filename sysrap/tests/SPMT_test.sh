#!/bin/bash -l 

REALDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
source $HOME/.opticks/GEOM/GEOM.sh 

name=SPMT_test
defarg="build_run_ana"
arg=${1:-$defarg}

FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

export FOLD 


vars="arg name REALDIR GEOM"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $REALDIR/$name.cc -I.. -std=c++11 -lstdc++ -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $REALDIR/$name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 


exit 0 

