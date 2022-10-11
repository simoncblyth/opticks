#!/bin/bash -l 
usage(){ cat << EOU
ct.sh : using CSGSimtraceTest
===============================

This uses the CSG intersection headers to get intersect positions using CUDA 
compatible code runing on the CPU, not the GPU where usually deployed.
The advantage of testing GPU code on CPU is ease of debugging.  

Workflow:

1. config GEOM with bash function geom_ OR vi ~/opticks/bin/GEOM_.sh 

2. when starting from Geant4 geometry must translate it into CSGFoundry geometry, 
   and persists into /tmp/$USER/opticks/GEOM/$GEOM/CSGFoundry with::

   ~/opticks/GeoChain/translate.sh 
   ~/opticks/CSG/ct.sh translate   # alternative to above  

3. run the CSG intersection, using CSGSimtraceTest::

   ~/opticks/CSG/ct.sh run
   ## persists intersects into /tmp/$USER/opticks/GEOM/$GEOM/CSGSimtraceTest

4. present the CSG intersections with python plots::

   ~/opticks/CSG/ct.sh ana

5. alternative way of doing steps 2,3 and 4 together::

   ~/opticks/CSG/ct.sh translate_run_ana
   
6. screencapture displayed plots into png::

   ~/opticks/CSG/ct.sh mpcap

7. copy captured png into publish repository for inclusion into presentations::

   PUB=example_distinguishing_string ~/opticks/CSG/ct.sh mppub

Control plot presentation with envvars::

   MPPLT_SIMTRACE_SELECTION_LINE=o2i ~/opticks/CSG/ct.sh ana

   GEOM=nmskSolidMaskTail__U1 FOCUS=-257,-39,7 ~/opticks/CSG/ct.sh ana  

   ct
   FOCUS=-257,-39,7  ./ct.sh ana   # show intersects from a portion of the geometry 


EOU
}

loglevels()
{
    export CSGSimtrace=INFO
}
#loglevels


arg=${1:-run_ana}

bin=CSGSimtraceTest
log=$bin.log

source $(dirname $BASH_SOURCE)/../bin/GEOM_.sh   # change the geometry with geom_ 
export FOLD=/tmp/$USER/opticks/GEOM/$GEOM/$bin/ALL

export TOPLINE="CSG/ct.sh GEOM $GEOM FOCUS $FOCUS"

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin GEOM FOLD"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/translate}" != "$arg" ]; then
    $(dirname $BASH_SOURCE)/../GeoChain/translate.sh 
    [ $? -ne 0 ] && echo $BASH_SOURCE translate error && exit 1 
fi  

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi  

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
        Darwin) lldb__ $bin ;;
        Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/ana}"  != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py
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

