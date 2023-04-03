#!/bin/bash -l 

source ../../bin/OPTICKS_INPUT_PHOTON.sh 

geom=V0J008
export GEOM=${GEOM:-$geom}
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


ipf=Hama:0:1000

#export OPTICKS_INPUT_PHOTON_FRAME=${OPTICKS_INPUT_PHOTON_FRAME:-$ipf}
export MOI=$ipf


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

