#!/bin/bash -l 

name=sblackbody_test
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

cd $(dirname $BASH_SOURCE)

defarg=info_build_run_ana
arg=${1:-$defarg}

vars="arg name FOLD bin"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -lm -I.. -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi 

exit 0 


