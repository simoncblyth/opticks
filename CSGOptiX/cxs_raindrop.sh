#!/bin/bash -l 
usage(){ cat << EOU
cxs_raindrop.sh : CXRaindropTest combining CFBASE_LOCAL raindrop geometry with standard CFBASE basis geometry  
=================================================================================================================

Grab from remote::

    cx
    ./tmp_grab.sh 

    ./cf_grab.sh    ## HUH: actually the tmp_grab.sh should be getting persisted CSGFoundry from within the GEOM dir 
  
EOU
}

export GEOM=BoxedSphere

case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac

msg="=== $BASH_SOURCE :"
arg=${1:-$argdef}

if [ "$(uname)" == "Darwin" ]; then 
    if [ "$arg" == "dru" -o "$arg" == "dan" ]; then  
        echo $msg dru or dan mode is local Darwin running and analysis 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain_Darwin/$GEOM
    else
        echo $msg run or ana mode handles Linux generated results grabbed from remote 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
    fi 
else
    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
fi 

unset GEOM                     # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc


bin=CXRaindropTest

if [ "${arg/run}" != "$arg" -o "${arg/dru}" != "$arg" ]; then 
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

if [ "${arg/ana}" != "$arg" -o "${arg/dan}" != "$arg" ]; then 

    #if [ "$(uname)" == "Darwin" ]; then
    #    opticks-switch-key remote   ## cx;cf_grab.sh to update local copy of the remote CSGFoundry for analysis consistency  
    #fi
    #  HMM NO SHOULD NOT USE THE CENTRALIZED STANDARD ONE : NEED TO USE THE SPECIFIC CSGFoundry FOR THIS TEST  

    export FOLD=$CFBASE_LOCAL/$bin
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py  
fi 


exit 0 
