#!/bin/bash 

usage(){ cat << EOU
SGLFW__DEPTH.sh 
=================

~/o/sysrap/tests/SGLFW__DEPTH.sh

EXE=CSGOptiXRenderInteractiveTest ~/o/sysrap/tests/SGLFW__DEPTH.sh


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

script=SGLFW__DEPTH.py 
defarg="info_ana"
vars="BASH_SOURCE PWD script arg defarg GEOM exe EXE NPY"

source $HOME/.opticks/GEOM/GEOM.sh 


exe=SGLFW_SOPTIX_Scene_test
#exe=CSGOptiXRenderInteractiveTest
EXE=${EXE:-$exe}

ls -1 $TMP/GEOM/$GEOM/$EXE/*_depth.npy
npy=$(ls -1 $TMP/GEOM/$GEOM/$EXE/*_depth.npy | tail -1)

export NPY=$npy

arg=${1:-$defarg}
if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${IPYTHON:-ipython} -i --pdb $script 
fi 



