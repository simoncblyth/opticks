#!/bin/bash -l 
usage(){ cat << EOU
CSGFoundrySimtraceRerunTest.sh
================================

::

    c
    ./CSGFoundrySimtraceRerunTest.sh 

EOU
}


arg=${1:-run}

bin=CSGFoundrySimtraceRerunTest
log=$bin.log
source $(dirname $BASH_SOURCE)/../bin/COMMON.sh 

UGEOMDIR=${GEOMDIR//$HOME\/}
BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")


T_FOLD=${FOLD/$bin/G4CXSimtraceTest}


if [ "info" == "$arg" ]; then
    vars="BASH_SOURCE arg bin GEOM GEOMDIR UGEOMDIR BASE UBASE FOLD T_FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 


if [ "run" == "$arg" ]; then
   [ -f "$log" ] && rm $log 
   export T_FOLD 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error $bin && exit 1 
fi 


exit 0 
