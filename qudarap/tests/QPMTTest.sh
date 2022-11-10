#!/bin/bash -l 

name=QPMTTest

defarg="run_ana"
arg=${1:-$defarg}

export QPMT=INFO
export FOLD=/tmp/$name

if [ "${arg/run}" != "$arg" ]; then  
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then  
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 
