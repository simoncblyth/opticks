#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
defarg=run_ana
arg=${1:-$defarg}
bin=stranTest

if [ "${arg/run}" != "$arg" ]; then 
    $bin 
    [ $? -ne 0 ] && echo $msg run error $bin && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=/tmp/$USER/opticks/$bin
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $msg ana error $bin && exit 2 
fi 

exit 0 

