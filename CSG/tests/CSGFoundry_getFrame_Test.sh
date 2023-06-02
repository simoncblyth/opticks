#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_getFrame_Test.sh
=============================

Loads a CSGFoundry geometry specified by GEOM envvar and 
accesses the frame specified by OPTICKS_INPUT_PHOTON_FRAME envvar.
The frame is used to transform some input photons and comparisons
are made between different transformation approaches. 

Examples::

    OPTICKS_INPUT_PHOTON_FRAME=PMT_20inch_veto:0:1000 ~/opticks/csg/tests/CSGFoundry_getFrame_Test.sh 


EOU
}


DIR=$(dirname $BASH_SOURCE)

source $DIR/../../bin/OPTICKS_INPUT_PHOTON.sh 

source ~/.opticks/GEOM/GEOM.sh  # sets GEOM envvar, edit with GEOM bash function

export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM


ipf=Hama:0:1000

export OPTICKS_INPUT_PHOTON_FRAME=${OPTICKS_INPUT_PHOTON_FRAME:-$ipf}
#export MOI=$ipf


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

    ${IPYTHON:-ipython} --pdb -i $DIR/$bin.py 
   [ $? -ne 0 ] && echo $msg ana error && exit 2
fi 

exit 0 

