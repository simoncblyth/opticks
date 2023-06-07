#!/bin/bash -l

name=SEvt__HasInputPhoton_Test

#export SEvt=INFO
defarg="run"
arg=${1:-$defarg}

export OPTICKS_INPUT_PHOTON=RainXZ100_f4.npy

vars="arg OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME"
for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/run}" != "$arg" ]; then 
   $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in
      Darwin) lldb__ $name ;;
      Linux)  gdb__ $name ;;  
   esac
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

exit 0 


