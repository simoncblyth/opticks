#!/bin/bash
usage(){ cat << EOU
ct.sh : using CSGSimtraceTest for CPU side testing intersect testing
======================================================================

This uses the CSG intersection headers to get intersect positions using CUDA
compatible code runing on the CPU, not the GPU where usually deployed.
The advantage of testing GPU code on CPU is ease of debugging.

Workflow:

1. config GEOM evvvar with bash function GEOM

2. the GEOM envvar and ${GEOM}_CFBaseFromGEOM evvar config where to load CSGFoundry from,
   the customary directory is::

    $HOME/.opticks/GEOM/$GEOM

3. run the CSG intersection, using CSGSimtraceTest::

   ~/opticks/CSG/ct.sh run
   LOG=1 ~/opticks/CSG/ct.sh run

   SOPR=0:0 ~/o/CSG/ct.sh     ## default first CSGPrim from first CSGSolid
   SOPR=1:0 ~/o/CSG/ct.sh     ## first prim from second CSGSolid, see CSGQuery

   SOPR=2:0 ~/o/CSG/ct.sh     ## check the first few CSGPrim in CSGSolid 2
   SOPR=2:1 ~/o/CSG/ct.sh
   SOPR=2:2 ~/o/CSG/ct.sh
   SOPR=2:3 ~/o/CSG/ct.sh
   SOPR=2:4 ~/o/CSG/ct.sh


   ## TODO: adopt MOI triplets for simpler prim selection


4. present the CSG intersections with python plots::

   ~/opticks/CSG/ct.sh ana

5. screencapture displayed plots into png::

   ~/opticks/CSG/ct.sh mpcap

6. copy captured png into publish repository for inclusion into presentations::

   PUB=example_distinguishing_string ~/opticks/CSG/ct.sh mppub


Control plot presentation with envvars::

   MPPLT_SIMTRACE_SELECTION_LINE=o2i ~/opticks/CSG/ct.sh ana

   GEOM=nmskSolidMaskTail__U1 FOCUS=-257,-39,7 ~/opticks/CSG/ct.sh ana

   ct
   FOCUS=-257,-39,7  ./ct.sh ana   # show intersects from a portion of the geometry


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))


logging()
{
    type $FUNCNAME
    export CSGSimtrace=INFO
    export SEvt__LIFECYCLE=INFO
    #export SEvt=INFO
}
[ -n "$LOG" ] && logging


defarg=info_run_ana
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}


bin=CSGSimtraceTest
log=$bin.log

source $HOME/.opticks/GEOM/GEOM.sh  # sets GEOM envvar
export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export FOLD=$TMP/GEOM/$GEOM/CSGSimtraceTest/ALL0_none/B000


export TOPLINE="CSG/ct.sh GEOM $GEOM FOCUS $FOCUS"

if [ "${arg/info}" != "$arg" ]; then
    vars="BASH_SOURCE arg bin GEOM ${GEOM}_CFBaseFromGEOM FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done
fi

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source ../bin/dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/ana}"  != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi

if [ "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=ct
    export CAP_STEM=${GEOM}
    case $arg in
       mpcap) source mpcap.sh cap  ;;
       mppub) source mpcap.sh env  ;;
    esac

    if [ "$arg" == "mppub" ]; then
        source epub.sh
    fi
fi

exit 0

