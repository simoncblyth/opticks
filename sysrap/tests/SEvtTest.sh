#!/bin/bash -l 

arg=${1:-run_ana}
msg="=== $BASH_SOURCE :"

if [ "${arg/run}" != "${arg}" ]; then 
    SEvtTest 
    [ $? -ne 0 ] && echo $msg run error && exit 1 
fi 

if [ "${arg/ana}" != "${arg}" ]; then 
    export FOLD=/tmp/$USER/opticks/SEvtTest 
    ${IPYTHON:-ipython} --pdb -i SEvtTest.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 2
fi

exit 0

