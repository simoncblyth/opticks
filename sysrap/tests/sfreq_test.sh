#!/bin/bash -l 

defarg="build_run_ana"
arg=${1:-$defarg}


name=sfreq_test 

export FOLD=/tmp/$name/out 
mkdir -p $FOLD


if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ -I.. -o /tmp/$name/$name 
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    /tmp/$name/$name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i sfreq_test.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 



exit 0 


