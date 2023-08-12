#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundry_CreateFromSimTest.sh testing CSGFoundry::CreateFromSim
=====================================================================

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
bin=CSGFoundry_CreateFromSimTest
script=$SDIR/CSGFoundryAB.py

source $HOME/.opticks/GEOM/GEOM.sh 


base=$HOME/.opticks/GEOM/$GEOM
#base=/tmp/GEOM/$GEOM
#base=/tmp/$USER/opticks/U4TreeCreateTest
export BASE=${BASE:-$base}

check=$BASE/CSGFoundry/SSim/stree/nds.npy
if [ ! -f "$check" ]; then
   echo $BASH_SOURCE input check $check does not exist at BASE $BASE check $check 
   exit 1 
fi 

export FOLD=/tmp/$USER/opticks/$bin
mkdir -p $FOLD

# env for CSGFoundryAB comparison 
export A_CFBASE=$BASE
export B_CFBASE=$FOLD

vars="BASH_SOURCE bin GEOM BASE FOLD check A_CFBASE B_CFBASE script"



loglevel(){
   export CSGFoundry=INFO
   #export CSGImport=INFO
   #export scsg_level=1
   lvid=119
   ndid=0
   export LVID=${LVID:-$lvid}
   export NDID=${NDID:-$ndid}
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

