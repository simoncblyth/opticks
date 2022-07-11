#!/bin/bash -l 

defarg=run_ana
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
    stranTest 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=/tmp/$USER/opticks/stranTest
    ${IPYTHON:-ipython} --pdb -i stranTest.py 
fi 

