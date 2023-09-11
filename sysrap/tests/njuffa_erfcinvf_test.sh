#!/bin/bash -l

usage(){ cat << EOU
njuffa_erfcinvf_test.sh
=========================


EOU
}

cd $(dirname $BASH_SOURCE) 
name=njuffa_erfcinvf_test

defarg="info_build_run_ana"
arg=${1:-$defarg}

FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name
export FOLD

vars="BASH_SOURCE name arg FOLD bin"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -I.. -std=c++11 -lstdc++ -I$HOME/np -o $bin
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

