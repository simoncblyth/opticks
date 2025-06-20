#!/bin/bash
usage(){ cat << EOU
SEvt_Lifecycle_Test.sh
=======================

::

   ~/opticks/sysrap/tests/SEvt_Lifecycle_Test.sh


DONE : get this to have some hits
--------------------------------------

Was failing to have hits due to the flagmask being EC|BT|TO
and using the default OPTICKS_HIT_MASK of SD.
Changing OPTICKS_HIT_MASK to EC yielded hits.

::

    In [5]: a
    Out[5]: SEvt symbol a pid -1 opt  off [0. 0. 0.] a.f.base /data1/blyth/tmp/GEOM/SEVT_LIFECYCLE_TEST/SEvt_Lifecycle_Test/ALL0_no_opticks_event_name/A000

    In [6]: a.fmtab
    Out[6]: array([['EC|BT|TO', '100']], dtype='<U21')



EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SEvt_Lifecycle_Test
bin=$name
script=$name.py

loglevel()
{
    echo $FUNCNAME : setting log levels
    export SEvt=INFO
}

[ -n "$LOG" ] && loglevel


defarg="run_ana"
arg=${1:-$defarg}


export GEOM=SEVT_LIFECYCLE_TEST

export OPTICKS_INPUT_PHOTON=RainXZ100_f4.npy
export OPTICKS_EVENT_MODE=DebugLite
export OPTICKS_MAX_BOUNCE=31
export OPTICKS_HIT_MASK=EC


evt=A000
tmp=/tmp/$USER/opticks
version=0

EVT=${EVT:-$evt}
TMP=${TMP:-$tmp}
VERSION=${VERSION:-$version}

export FOLD=$TMP/GEOM/$GEOM/$bin/ALL${VERSION}_no_opticks_event_name/$EVT

logging()
{
   type $FUNCNAME
   export SEvt__SAVE=1
}
[ -n "$LOG" ] && logging


vars="arg name bin script OPTICKS_INPUT_PHOTON OPTICKS_EVENT_MODE EVT TMP GEOM VERSION FOLD"


if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi

exit 0


