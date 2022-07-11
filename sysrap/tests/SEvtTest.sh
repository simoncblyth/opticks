#!/bin/bash -l 

source ../../bin/GEOM_.sh 
source ../../bin/OPTICKS_INPUT_PHOTON.sh 

export OPTICKS_INPUT_PHOTON_FRAME=0 
export CFBASE 

defarg=run_ana
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
    SEvtTest 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=/tmp/$USER/opticks/SEvtTest/test_InputPhoton
    ${IPYTHON:-ipython} --pdb -i SEvtTestIP.py 
fi 


