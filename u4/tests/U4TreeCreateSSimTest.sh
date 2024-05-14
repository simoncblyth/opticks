#!/bin/bash -l 
usage(){ cat << EOU
U4TreeCreateSSimTest.sh  : loads GEOM configured geometry, runs U4Tree::Create populating SSim/stree.h, saves to FOLD  
==============================================================================================================================

::

    GEOM # edit according to geometry source 
   ~/o/u4/tests/U4TreeCreateSSimTest.sh 

EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

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

#export SSim__stree_level=0 
#export sn__uncoincide_dump_lvid=107 
#export sn__uncoincide_dump_lvid=106


#export U4Tree__DISABLE_OSUR_IMPLICIT=1   ## TEMPORAILY TO SEE IF OSUR EXPLAINS ALL BOUNDARY DEVIATION
#export U4Polycone__DISABLE_NUDGE=1 



source $HOME/.opticks/GEOM/GEOM.sh
[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM must be defined && exit 1  

gdmlpath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

if [ -f "$gdmlpath" ]; then
   export ${GEOM}_GDMLPath=$gdmlpath
else
   echo $BASH_SOURCE : GEOM $GEOM LACKS gdmlpath $gdmlpath : ASSUME USING ANOTHER GEOMETRY SOURCE APPROACH  
fi 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
fold=$TMP/$bin/$GEOM
mkdir -p $fold

export FOLD=$fold   # where SSim saved/loaded 

U4TreeCreateSSimTest


script=$SDIR/$bin.py 

vars="BASH_SOURCE SDIR bin GEOM gdmlpath tmp TMP FOLD script"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 


if [ "${arg/clean}" != "$arg" ]; then 
    cd $TMP && rm -rf U4TreeCreateSSimTest  # hardcode directory name for safety 
    [ $? -ne 0 ] && echo $BASH_SOURCE clean error && exit 1 
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
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

