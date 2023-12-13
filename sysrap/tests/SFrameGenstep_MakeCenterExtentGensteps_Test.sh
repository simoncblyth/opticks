#!/bin/bash -l 
usage(){ cat << EOU
SFrameGenstep_MakeCenterExtentGensteps_Test.sh
===============================================

::


   ~/o/sysrap/tests/SFrameGenstep_MakeCenterExtentGensteps_Test.sh

EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

defarg="info_run_ana"
arg=${1:-$defarg}
name=SFrameGenstep_MakeCenterExtentGensteps_Test
script=$SDIR/$name.py 

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}

export FOLD=$TMP/$name
mkdir -p $FOLD

vars="BASH_SOURCE SDIR TMP FOLD"


cehigh(){
  export CEHIGH_0=-11:-9:0:0:-3:-1:100:2
  export CEHIGH_1=9:11:0:0:-3:-1:100:2  
  export CEHIGH_2=-1:1:0:0:-3:-1:100:2
}

[ -n "$CEHIGH" ] && cehigh 


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $name
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

