#!/bin/bash -l 
usage(){ cat << EOU
U4TreeCreateSSimTest.sh  : loads GDML, runs U4Tree::Create populating SSim/stree.h, saves to FOLD  
====================================================================================================

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

bin=U4TreeCreateSSimTest 
defarg="info_run_ana"
arg=${1:-$defarg}

loglevels(){
   #export U4VolumeMaker=INFO
   #export U4Solid=INFO
   export DUMMY=INFO
}
loglevels


#export U4TreeBorder__FLAGGED_ISOLID=HamamatsuR12860sMask_virtual0x61b0510
#export U4Tree__IsFlaggedSolid_NAME=HamamatsuR12860sMask_virtual

export SSim__stree_level=0 
#export sn__uncoincide_dump_lvid=107 
export sn__uncoincide_dump_lvid=106

#export U4Polycone__DISABLE_NUDGE=1 



source $HOME/.opticks/GEOM/GEOM.sh
gdmlpath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

if [ ! -f "$gdmlpath" ]; then
   echo $BASH_SOURCE : ERROR GEOM $GEOM LACKS gdmlpath $gdmlpath 
   exit 1 
fi 
export ${GEOM}_GDMLPath=$gdmlpath


fold=/tmp/$USER/opticks/$bin

export BASE=$fold   # where SSim loaded from when executable has any argumnent
export FOLD=$fold   # where SSim saved 

script=$SDIR/$bin.py 

vars="BASH_SOURCE SDIR bin GEOM gdmlpath BASE FOLD script"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    echo $BASH_SOURCE : FATAL bin $bin IS BUILD BY STANDARD u4 OM : NOT THIS SCRIPT && exit 1
fi 

if [ "${arg/load}" != "$arg" ]; then 
    $bin load
    [ $? -ne 0 ] && echo $BASH_SOURCE load error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in
    Darwin) lldb__ $bin ;;
    Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

