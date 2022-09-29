#!/bin/bash -l 

name=U4Hit_Debug_Test

export FOLD=/tmp

$name
[ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

${IPYTHON:-ipython} --pdb -i $name.py 
[ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 


exit 0


