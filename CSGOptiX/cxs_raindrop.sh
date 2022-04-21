#!/bin/bash -l 
usage(){ cat << EOU
cxs_raindrop.sh : CXRaindropTest combining CFBASE_LOCAL raindrop geometry with standard CFBASE basis geometry  
=================================================================================================================

Grab from remote::

    cx
    ./tmp_grab.sh 
    ./cf_grab.sh 
  
EOU
}

export GEOM=BoxedSphere

if [ "$(uname)" == "Darwin" ]; then 

    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain_Darwin/$GEOM
    #export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM

    # HMM: when analysing grabbed outputs from Linux on Darwin need to use the Linux dir
else
    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
fi 


unset GEOM   # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc

msg="=== $BASH_SOURCE :"

case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac
arg=${1:-$argdef}

bin=CXRaindropTest

if [ "${arg/run}" != "$arg" ]; then 
    logdir=/tmp/$USER/opticks/$bin
    mkdir -p $logdir
    iwd=$PWD
    cd $logdir

    if [ -n "$DEBUG" ]; then 
        lldb__  $bin
    else
        $bin
    fi 

    [ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 
    echo $msg logdir $logdir 
    cd $iwd
fi 

if [ "${arg/ana}" != "$arg" ]; then 

    if [ "$(uname)" == "Darwin" ]; then
        opticks-switch-key remote   ## cx;cf_grab.sh to update local copy of the remote CSGFoundry for analysis consistency  
    fi 
    export FOLD=$CFBASE_LOCAL/$bin
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py  
fi 


exit 0 
