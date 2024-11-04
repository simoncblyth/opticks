#!/bin/bash 
usage(){ cat << EOU
U4TreeCreateTest.sh  : loads geometry, runs U4Tree::Create populating stree.h, saves to FOLD  
==============================================================================================

The geometry may be loaded by various means including from GDML or j/PMTSim via U4VolumeMaker::

    ~/o/u4/tests/U4TreeCreateTest.sh

The test U4Polycone_test.sh avoids slow rebuild after changing WITH macros
by directly compiling the needed SysRap sources and not using 
the full SysRap library.  

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

bin=U4TreeCreateTest 
defarg="info_run_ana"
arg=${1:-$defarg}

loglevels(){
   #export U4VolumeMaker=INFO
   #export U4Solid=INFO
   export DUMMY=INFO
}

loglevels


#export U4TreeBorder__FLAGGED_ISOLID=HamamatsuR12860sMask_virtual0x61b0510
export U4Solid__IsFlaggedLVID=0
#export U4Solid__IsFlaggedName=sDeadWater

source $HOME/.opticks/GEOM/GEOM.sh
gdmlpath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

TMP=${TMP:-/tmp/$USER/opticks}
export FOLD=$TMP/$bin
script=$SDIR/$bin.py 

vars="BASH_SOURCE SDIR bin GEOM gdmlpath FOLD script"


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/clean}" != "$arg" ]; then 
    cd $TMP && rm -rf U4TreeCreateTest  # hardcode directory name for safety 
    [ $? -ne 0 ] && echo $BASH_SOURCE clean error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 


    if [ -f "$gdmlpath" ]; then 
       echo $BASH_SOURCE : GEOM $GEOM has gdmlpath $gdmlpath setting ${GEOM}_GDMLPath
       export ${GEOM}_GDMLPath=$gdmlpath
    else
       echo $BASH_SOURCE : NOTE GEOM $GEOM LACKS gdmlpath ASSUME USING non-gdml  U4VolumeMaker::PV resolution approach 
    fi 

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


