#!/bin/bash
usage(){ cat << EOU
QEventTest.sh
=============

::
   
   ~/o/qudarap/tests/QEventTest.sh 
   ~/o/qudarap/tests/QEventTest.sh dbg

   TEST=one ~/o/qudarap/tests/QEventTest.sh 
   BP=cudaMalloc ~/o/qudarap/tests/QEventTest.sh 


Simple way to check for GPU memory leaks while running 
a QEventTest is to run nvidia-smi in another window::

    nvidia-smi -lms 500    # every half second  

Fancier way is to use ~/o/sysrap/smonitor.sh 
to collect a memory profile into NumPy array
for plotting. 

EOU
}
cd $(dirname $(realpath $BASH_SOURCE))
source dbg__.sh 


export GEOM=DummyGEOMForQEventTest

name=QEventTest 

#test=many
#test=ALL
test=loaded

export TEST=${TEST:-$test}
script0=$name.py 
script1=${name}_${TEST}.py 

#export OPTICKS_NUM_EVENT=1000 
#export OPTICKS_NUM_EVENT=100
export OPTICKS_NUM_EVENT=10

logging(){
   type $FUNCNAME
   export QEvent=INFO
   export QEvent__LIFECYCLE=1
}
[ -n "$LOG" ] && logging

defarg="info_run"
[ -n "$BP" ] && defarg="info_dbg"
arg=${1:-$defarg}


vars="BASH_SOURCE 0 PWD name test TEST defarg arg BP OPTICKS_NUM_EVENT LOG"

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

