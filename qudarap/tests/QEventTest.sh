#!/bin/bash -l 
usage(){ cat << EOU
QEventTest.sh
=============

::
   
   ~/o/qudarap/tests/QEventTest.sh 
   BP=cudaMalloc ~/o/qudarap/tests/QEventTest.sh 


Simple way to check for GPU memory leaks while running 
a QEventTest is to run nvidia-smi in another window::

    nvidia-smi -lms 500    # every half second  


EOU
}


name=QEventTest 

export TEST=setGenstep_many 
export OPTICKS_NUM_EVENT=1000 
export QEvent__LIFECYCLE=1

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

