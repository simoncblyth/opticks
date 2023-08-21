#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_CreateFromSimTest.sh testing CSGFoundry::CreateFromSim
=====================================================================

HMM: problem with this is that there is only one CSGFoundry 
so cannot compare between two routes. This is becaise when using 
U4TreeCreate start from gdml and create SSim that creates the CSGFoundry. 


So will need to go upwards to gxt/G4CXOpticks_setGeometry_Test.sh

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
bin=CSGFoundry_CreateFromSimTest
script=$SDIR/CSGFoundryAB.py

source $HOME/.opticks/GEOM/GEOM.sh 


#base=$HOME/.opticks/GEOM/$GEOM
#base=/tmp/GEOM/$GEOM
#base=/tmp/$USER/opticks/U4TreeCreateTest
base=/tmp/blyth/opticks/U4TreeCreateSSimTest

export BASE=${BASE:-$base}

check=$BASE/SSim/stree/nds.npy

if [ ! -f "$check" ]; then
   echo $BASH_SOURCE input check $check does not exist at BASE $BASE check $check 
   exit 1 
fi 

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

# env for CSGFoundryAB comparison 
export A_CFBASE=/tmp/$USER/opticks/G4CXOpticks_setGeometry_Test
export B_CFBASE=$FOLD

lvid=119
ndid=0
export LVID=${LVID:-$lvid}
export NDID=${NDID:-$ndid}
#export scsg_level=1

# this does nothing here as no U4 involved 
#export U4Polycone__DISABLE_NUDGE=1 




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
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__  $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 1 
fi 

exit 0

