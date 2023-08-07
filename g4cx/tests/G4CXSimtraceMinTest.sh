#!/bin/bash -l 
usage(){ cat << EOU
G4CXSimtraceMinTest.sh 
======================

Using outputs from G4CXAppTest.sh 

::

   MODE=2 APID=0 ./G4CXSimtraceMinTest.sh 


EOU
}
SDIR=$(cd $(dirname $BASH_SOURCE) && pwd )
name=G4CXSimtraceMinTest
script=$SDIR/$name.py 
source $HOME/.opticks/GEOM/GEOM.sh 


export VERSION=0
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/G4CXAppTest
export EVT=001
export AFOLD=$BASE/ALL${VERSION}/p${EVT}
export BFOLD=$BASE/ALL${VERSION}/n${EVT}
export TFOLD=$BASE/0/p999

vars="BASH_SOURCE name SDIR GEOM BASE AFOLD BFOLD TFOLD script"

defarg="info_ana"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script

fi 

