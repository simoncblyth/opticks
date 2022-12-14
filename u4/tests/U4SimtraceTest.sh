#!/bin/bash -l 
usage(){ cat << EOU
U4SimtraceTest.sh
==========================

::

    N=0 ./U4SimtraceTest.sh 
    N=1 ./U4SimtraceTest.sh 

    APID=173 PLAB=1 BGC=yellow ./U4SimtraceTest.sh ana

Z-changing big bouncers::

    N=0 APID=256 PLAB=1 BGC=white ./U4SimtraceTest.sh ana
    N=1 APID=261 PLAB=1 BGC=white ./U4SimtraceTest.sh ana

Grab the custom boundary status for each point::

    In [25]: t.aux[261,:32,1,3].copy().view(np.int8)[::4].copy().view("|S32")
    Out[25]: array([b'TTTZNZRZNZA'], dtype='|S32')


::


    N=0 APID=726 BPID=-1 AOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana

    N=1 APID=-1 BPID=726 BOPT=nrm FOCUS=0,10,185 ./U4SimtraceTest.sh ana



EOU
}

bin=U4SimtraceTest

export GEOM=hamaLogicalPMT
export GEOMFOLD=/tmp/$USER/opticks/GEOM/$GEOM
export BASE=$GEOMFOLD/$bin

export VERSION=${N:-0}

#export LAYOUT=two_pmt
export LAYOUT=one_pmt

export FOLD=$BASE/$VERSION


export AFOLD=$GEOMFOLD/U4SimulateTest/ALL0
export APID=${APID:-0}

export BFOLD=$GEOMFOLD/U4SimulateTest/SEL1
export BPID=${BPID:-0}

geomscript=$GEOM.sh 

if [ -f "$geomscript" ]; then  
    source $geomscript $VERSION $LAYOUT
else
    echo $BASH_SOURCE : no gemoscript $geomscript
fi 

# python ana level presentation 
export LOC=skip


loglevels()
{
    export U4VolumeMaker=INFO
}
loglevels


log=${bin}.log
logN=${bin}_$VERSION.log

defarg="run_ana"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "${arg/ana}" != "$arg"  ]; then
    [ "$arg" == "nana" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 

