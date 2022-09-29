#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
name=U4Debug_Test
fold=/tmp/$name

export U4Debug_SaveDir=$fold
export FOLD0=$fold/000
export FOLD1=$fold/001

export U4Scintillation_Debug=INFO
export U4Cerenkov_Debug=INFO
export U4Hit_Debug=INFO
export U4Debug=INFO


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

