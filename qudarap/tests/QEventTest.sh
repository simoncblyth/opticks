#!/bin/bash -l 
usage(){ cat << EOU

::
   
   ~/o/qudarap/tests/QEventTest.sh 
   BP=cudaMalloc ~/o/qudarap/tests/QEventTest.sh 


EOU
}


name=QEventTest 

export TEST=setGenstep_many 
export OPTICKS_NUM_EVENT=1000 

defarg="run"
[ -n "$BP" ] && defarg="dbg"

arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
   $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
   dbg__ $name 
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

exit 0 

