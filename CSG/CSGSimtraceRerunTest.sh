#!/bin/bash -l 
usage(){ cat << EOU
CSGSimtraceRerunTest.sh
================================

CSGSimtraceRerunTest requires a CSGFoundry geometry and corresponding 
simtrace intersect array. Create these with:: 

    geom_ ## set geom to "nmskSolidMask" :  a string understood by PMTSim::getSolid

    gc     ## cd ~/opticks/GeoChain
    ./translate.sh    ## translate PMTSim::getSolid Geant4 solid into CSGFoundry 

    gx     ## cd ~/opticks/g4cx 

    ./gxt.sh        # simtrace GPU run on workstation
    ./gxt.sh grab   # rsync back to laptop
    ./gxt.sh ana    # python plotting  



SELECTION envvar 
    provides simtrace indices to rerun on CPU using CSGQuery which runs
    the CUDA compatible intersect code on the CPU. 
    Without this envvar all the simtrace items ate rerun.

    When using a rerun selection of a few intersects only it is 
    possible to switch on full debug verbosity after 
    recompilinh the CSG package with two non-standard preprocessor macros, 
    that are not intended to ever be committed::

        DEBUG
        DEBUG_RECORD





::

    c
    ./CSGSimtraceRerunTest.sh 
    ./CSGSimtraceRerunTest.sh info
    ./CSGSimtraceRerunTest.sh run 
    ./CSGSimtraceRerunTest.sh ana

EOU
}


arg=${1:-run}

bin=CSGSimtraceRerunTest
log=$bin.log
source $(dirname $BASH_SOURCE)/../bin/COMMON.sh 

UGEOMDIR=${GEOMDIR//$HOME\/}
BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")


export CSGFoundry=INFO


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

if [ "dbg" == "$arg" ]; then
   [ -f "$log" ] && rm $log 
   export T_FOLD 

   case $(uname) in 
      Darwin) lldb__ $bin ;;
      Linux) gdb__ $bin ;;
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error $bin && exit 1 
fi 

if [ "ana" == "$arg" ]; then
   export T_FOLD 
   ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error $bin && exit 2 
fi 


exit 0 
