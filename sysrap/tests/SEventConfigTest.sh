#!/bin/bash -l

name=SEventConfigTest

export OPTICKS_EVENTMODE=StandardFullDebug

#export OPTICKS_OUT_FOLD=/tmp/$USER/opticks/SEventConfigTest/out_fold
#export OPTICKS_OUT_NAME=oganized/relative/dir/tree/out_name

export OPTICKS_INPUT_PHOTON=/some/path/to/name.npy 
export OPTICKS_INPUT_PHOTON_FRAME=Hama:0:1000




export BASE=/tmp/blyth/opticks/J003/G4CXSimulateTest
export FOLD=/tmp/$USER/opticks/$name

defarg="run"
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in
       Darwin) lldb__ $name ;;
       Linux)  gdb__  $name ;;
   esac     
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

exit 0 


