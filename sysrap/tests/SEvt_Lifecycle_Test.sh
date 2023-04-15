#!/bin/bash -l

name=SEvt_Lifecycle_Test

loglevel()
{
    echo $FUNCNAME : setting log levels 
    export SEvt=INFO
}

[ -n "$DBG" ] && loglevel 


defarg="run_ana"
arg=${1:-$defarg}

export OPTICKS_INPUT_PHOTON=RainXZ100_f4.npy
export OPTICKS_EVENT_MODE=StandardFullDebug

evt=001
export EVT=${EVT:-$evt}
export FOLD=/tmp/$USER/opticks/GEOM/$name/ALL/$EVT

vars="arg OPTICKS_INPUT_PHOTON OPTICKS_EVENT_MODE EVT FOLD"
for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/run}" != "$arg" ]; then 
   $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py  
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 


