#!/bin/bash -l
usage(){ cat << EOU
SEvt_Lifecycle_Test.sh
=======================

::
 
   ~/opticks/sysrap/tests/SEvt_Lifecycle_Test.sh


EOU
}

name=SEvt_Lifecycle_Test

loglevel()
{
    echo $FUNCNAME : setting log levels 
    export SEvt=INFO
}

[ -n "$LOG" ] && loglevel 


defarg="run_ana"
arg=${1:-$defarg}

cd $(dirname $BASH_SOURCE)

export GEOM=SEVT_LIFECYCLE_TEST

export OPTICKS_INPUT_PHOTON=RainXZ100_f4.npy
export OPTICKS_EVENT_MODE=StandardFullDebug
export OPTICKS_MAX_BOUNCE=31 

evt=p001
tmp=/tmp/$USER/opticks
version=0 

EVT=${EVT:-$evt}
TMP=${TMP:-$tmp}
VERSION=${VERSION:-$version}

export FOLD=$TMP/GEOM/$GEOM/$name/ALL$VERSION/$EVT


vars="arg OPTICKS_INPUT_PHOTON OPTICKS_EVENT_MODE EVT TMP GEOM VERSION FOLD"
for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 

if [ "${arg/run}" != "$arg" ]; then 
   $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $name
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py  
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

exit 0 


