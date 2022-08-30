#!/bin/bash -l 

defarg="run"
arg=${1:-$defarg}
bin=SFrameGenstep_MakeCenterExtentGensteps_Test

fold=/tmp/$USER/opticks/$bin
mkdir -p $fold

export FOLD=$fold
export CEHIGH_0=-11:-9:0:0:-3:-1:100:2

if [ "run" == "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1  
fi 

if [ "ana" == "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/$bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 


exit 0 


