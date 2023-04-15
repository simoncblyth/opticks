#!/bin/bash -l

name=SEvt_Lifecycle_Test

loglevel()
{
    echo $FUNCNAME : setting log levels 
    export SEvt=INFO
}

[ -n "$DBG" ] && loglevel 



export OPTICKS_INPUT_PHOTON=RainXZ100_f4.npy
export OPTICKS_EVENT_MODE=StandardFullDebug


$name 




