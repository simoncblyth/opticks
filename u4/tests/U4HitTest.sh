#!/bin/bash -l 

export SOpticksResource_ExecutableName=G4CXSimulateTest
source $(dirname $BASH_SOURCE)/../../bin/COMMON.sh

bin=U4HitTest 
msg="=== $BASH_SOURCE :"

#defarg="run_ana"
defarg="run"
arg=${1:-$defarg}

export SEvt=info

if [ "${arg/run}" != "$arg" ]; then 
   $bin
   [ $? -ne 0 ] && echo $msg run $bin error && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   case $(uname) in
     Linux) gdb__  $bin ;;
     Darwin) lldb__  $bin ;;
   esac
   [ $? -ne 0 ] && echo $msg dbg $bin error && exit 2
fi 


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $bin.py 
   [ $? -ne 0 ] && echo $msg ana error && exit 3
fi 

exit 0 




