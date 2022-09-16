#!/bin/bash -l 
usage(){ cat << EOU
CSGSimtraceSampleTest.sh
================================

CSGSimtraceSampleTest requires a CSGFoundry geometry and corresponding 
small simtrace intersect array. Create these with:: 

    geom_ ## set geom to "nmskSolidMask" :  a string understood by PMTSim::getSolid

    gc     ## cd ~/opticks/GeoChain
    ./translate.sh    ## translate PMTSim::getSolid Geant4 solid into CSGFoundry 

EOU
}


arg=${1:-run}

bin=CSGSimtraceSampleTest
log=$bin.log
source $(dirname $BASH_SOURCE)/../bin/COMMON.sh 

UGEOMDIR=${GEOMDIR//$HOME\/}
BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")

export CSGFoundry=INFO
export CSGSimtraceSample=INFO 

export SAMPLE_PATH=/tmp/simtrace_sample.npy

if [ "info" == "$arg" ]; then
    vars="BASH_SOURCE arg bin GEOM GEOMDIR UGEOMDIR BASE SAMPLE_PATH"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi 

if [ "run" == "$arg" ]; then
   [ -f "$log" ] && rm $log 
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error $bin && exit 1 
fi 

if [ "dbg" == "$arg" ]; then
   [ -f "$log" ] && rm $log 
   case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux) gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error $bin && exit 1 
fi 

if [ "ana" == "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error $bin && exit 2 
fi 

exit 0 
