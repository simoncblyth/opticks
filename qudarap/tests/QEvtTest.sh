#!/bin/bash
usage(){ cat << EOU
QEvtTest.sh
=============

::

   ~/o/qudarap/tests/QEvtTest.sh
   ~/o/qudarap/tests/QEvtTest.sh dbg

   TEST=one ~/o/qudarap/tests/QEvtTest.sh
   BP=cudaMalloc ~/o/qudarap/tests/QEvtTest.sh


Simple way to check for GPU memory leaks while running
a QEvtTest is to run nvidia-smi in another window::

    nvidia-smi -lms 500    # every half second

Fancier way is to use ~/o/sysrap/smonitor.sh
to collect a memory profile into NumPy array
for plotting.

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh


export GEOM=DummyGEOMForQEvtTest

name=QEvtTest



#test=many
#test=ALL
#test=loaded


test=PerLaunchMerge


#test=LiteFinalMerge
#test=LiteFinalMerge_async
#test=FullFinalMerge
#test=FullFinalMerge_async

export TEST=${TEST:-$test}
script0=$name.py
script1=${name}_${TEST}.py


tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
export FOLD=$TMP/$name/$TEST
mkdir -p $FOLD


#export OPTICKS_NUM_EVENT=1000
#export OPTICKS_NUM_EVENT=100
export OPTICKS_NUM_EVENT=10

case $TEST in
    LiteFinalMerge|LiteFinalMerge_async|FullFinalMerge|FullFinalMerge_async|PerLaunchMerge)
        unset OPTICKS_HIT_MASK
        export OPTICKS_HIT_MASK=EC
        export OPTICKS_MERGE_WINDOW=1
        ;;
esac





logging(){
   type $FUNCNAME
   export QEvt=INFO
   export QEvt__LIFECYCLE=1
}
[ -n "$LOG" ] && logging

defarg="info_run"
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}



# TEST=merge_M10 cxs_min.sh
afold=/data1/blyth/tmp/GEOM/J25_4_0_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_merge_M10/A000
# OJ running ?  TEST=hitlite/hitlitemerged ojt
#afold=/tmp/blyth/opticks/GEOM/J25_4_0_opticks_Debug/python3.11/ALL0_no_opticks_event_name/A000
export AFOLD=${AFOLD:-$afold}


vars="BASH_SOURCE 0 PWD name test TEST FOLD defarg arg BP OPTICKS_NUM_EVENT LOG AFOLD"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/pdb0}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i  $script0
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb0 error && exit 2
fi

if [ "${arg/pdb1}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i  $script1
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb1 error && exit 2
fi



exit 0

