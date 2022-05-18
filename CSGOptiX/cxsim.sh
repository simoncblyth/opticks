#!/bin/bash -l 
source $PWD/../bin/GEOM.sh trim   ## sets GEOM envvar based on GEOM.txt file 

msg="=== $BASH_SOURCE :"

case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac
arg=${1:-$argdef}

usage(){ cat << EOU
cxsim.sh : CSGOptiXSimulateTest combining CFBASE_LOCAL simple test geometry with standard CFBASE basis geometry  
=================================================================================================================

Create the geometry, at CSG level::

   c
   ./CSGMakerTest.sh   
   ## TODO: confusingly this writes into GeoChain directory, switch to CSGMakerTest     
   ## OR change to a more appropriately named location for test geometries

OR translate from G4 using the GeoChain::

   gc
   ./translate.sh 

Run the sim::

   cx
   ./cxsim.sh 

Grab from remote::

    cx
    ./tmp_grab.sh 
    ./cf_grab.sh 


Laptop analysis::

   cx
   ./cxsim.sh ana
  
EOU
}

export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
unset GEOM   # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then 
    logdir=/tmp/$USER/opticks/CSGOptiXSimulateTest 
    mkdir -p $logdir
    iwd=$PWD
    cd $logdir

    if [ "${arg/run}" != "$arg" ] ; then
        CSGOptiXSimulateTest 
    elif [ "${arg/dbg}" != "$arg" ] ; then
        gdb CSGOptiXSimulateTest 
    fi  

    [ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 
    echo $msg logdir $logdir 
    cd $iwd
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    if [ "$(uname)" == "Darwin" ]; then
        opticks-switch-key remote   ## cx;cf_grab.sh to update local copy of the remote CSGFoundry for analysis consistency  
    fi 
    export FOLD=$CFBASE_LOCAL/CSGOptiXSimulateTest
    ${IPYTHON:-ipython} --pdb -i tests/CSGOptiXSimulateTest.py  
fi 


exit 0 
