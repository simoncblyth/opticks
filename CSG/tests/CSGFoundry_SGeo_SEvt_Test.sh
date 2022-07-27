#!/bin/bash -l 


msg="=== $BASH_SOURCE :"

bin=CSGFoundry_SGeo_SEvt_Test
export SOpticksResource_ExecutableName=G4CXSimulateTest
source $(dirname $BASH_SOURCE)/../../bin/COMMON.sh 

#defarg="run_ana"
defarg="run"
arg=${1:-$defarg}

export SEvt=info

export FOLD=/tmp/$USER/opticks/$bin

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $msg run $bin error && exit 1
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    ${IPYTHON:-ipython} --pdb -i $bin.py 
   [ $? -ne 0 ] && echo $msg ana error && exit 2
fi 

exit 0 

