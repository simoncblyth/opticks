#!/bin/bash -l 

bin=U4TreeTest 
defarg="run_ana"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=/tmp/$USER/opticks/U4TreeTest
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

exit 0 


