#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=U4Scintillation_Debug_Test
fold=/tmp/$name

export U4Scintillation_Debug_SaveDir=$fold
export FOLD0=$fold/z000
export FOLD1=$fold/p001

export U4Scintillation_Debug=INFO


arg=${1:-run_ana}

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $msg run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $msg ana error && exit 2 
fi 

exit 0 

