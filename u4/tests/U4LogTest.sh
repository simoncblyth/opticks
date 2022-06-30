#!/bin/bash -l 

defarg="run_ana"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then  
   U4LogTest 
   [ $? -ne 0 ] && echo run error *&& exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then  
   ${IPYTHON:-ipython} --pdb -i U4LogTest.py 
   [ $? -ne 0 ] && echo ana error *&& exit 2 
fi 

exit 0 


