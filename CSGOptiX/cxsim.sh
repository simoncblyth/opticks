#!/bin/bash -l 
msg="=== $BASH_SOURCE :"
case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac
arg=${1:-$argdef}
bin=CSGOptiXSimTest

usage(){ cat << EOU
cxsim.sh : $bin : standard geometry and SSim inputs 
===============================================================

Create the standard geometry::

   cg
   ./run.sh 


Run the sim::

   cx
   ./cxsim.sh 

Grab from remote::

    cx
    #?
    #./tmp_grab.sh 
    #./cf_grab.sh 


Laptop analysis::

   cx
   ./cxsim.sh ana
  
EOU
}

export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then 
    logdir=/tmp/$USER/opticks/$bin
    mkdir -p $logdir
    iwd=$PWD
    cd $logdir

    if [ "${arg/run}" != "$arg" ] ; then
        $bin
    elif [ "${arg/dbg}" != "$arg" ] ; then
        gdb $bin
    fi  

    [ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 
    echo $msg logdir $logdir 
    cd $iwd
fi 

#if [ "${arg/ana}" != "$arg" ]; then 
    #${IPYTHON:-ipython} --pdb -i tests/$bin.py  
#fi 


exit 0 
