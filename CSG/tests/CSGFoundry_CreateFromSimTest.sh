#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_CreateFromSimTest.sh testing CSGFoundry::CreateFromSim
=====================================================================

This is a technical test that are able to copy one CSGFoundry 
to another, rather than a compatison of two routes. 
This is because when using U4TreeCreate start from gdml 
and create SSim that creates the CSGFoundry. 
So will need to go upwards to gxt/G4CXOpticks_setGeometry_Test.sh
for a more complete test. 

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
bin=CSGFoundry_CreateFromSimTest

#script=$SDIR/$bin.py
script=$SDIR/CSGFoundryAB.py

source $HOME/.opticks/GEOM/GEOM.sh 

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

# env for CSGFoundryAB comparison 
export A_CFBASE=$HOME/.opticks/GEOM/$GEOM
export B_CFBASE=$FOLD

lvid=119
ndid=0
export LVID=${LVID:-$lvid}
export NDID=${NDID:-$ndid}
#export scsg_level=1




vars="BASH_SOURCE bin GEOM BASE FOLD check A_CFBASE B_CFBASE script LVID NDID scsg_level"

loglevel(){
   export CSGFoundry=INFO
   #export CSGImport=INFO
}
loglevel


defarg=info_run_ana
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0

