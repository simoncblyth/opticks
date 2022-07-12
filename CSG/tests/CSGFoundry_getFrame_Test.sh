#!/bin/bash -l 

source ../../bin/OPTICKS_INPUT_PHOTON.sh 

export OPTICKS_INPUT_PHOTON_FRAME="Hama:0:1000"


msg="=== $BASH_SOURCE :"
bin=CSGFoundry_getFrame_Test
defarg="run_ana"
arg=${1:-$defarg}

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

