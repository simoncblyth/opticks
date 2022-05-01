#!/bin/bash -l 
usage(){ cat << EOU
cxs_raindrop.sh : CXRaindropTest combining CFBASE_LOCAL raindrop geometry with standard CFBASE basis geometry  
=================================================================================================================

Grab from remote::

    cx
    ./tmp_grab.sh 

    ## ./cf_grab.sh    
    ##     actually the tmp_grab.sh should be getting the specific persisted CSGFoundry 
    ##     should not need the remote central one  ?

    PIDX=0 ./cxs_raindrop.sh  
  
EOU
}

export GEOM=BoxedSphere

case $(uname) in 
   Linux)  argdef=run  ;;
   Darwin) argdef=ana  ;;
esac

msg="=== $BASH_SOURCE :"
arg=${1:-$argdef}
bin=CXRaindropTest

if [ "$(uname)" == "Darwin" ]; then 
    if [ "$arg" == "dru" -o "$arg" == "dan" ]; then  
        echo $msg dru or dan mode is local Darwin running and analysis 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain_Darwin/$GEOM
        export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/$(SCVDLabel)/$(CSGOptiXVersion)
    else
        echo $msg run or ana mode handles Linux generated results grabbed from remote 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
        export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/SCVD0/70000
    fi 
else
    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
    export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/$(SCVDLabel)/$(CSGOptiXVersion)
fi 


vars="arg bin GEOM CFBASE_LOCAL OPTICKS_OUT_FOLD FOLD"
dumpvars(){ for var in $vars ; do printf "%25s : %s \n" $var ${!var} ; done ; }
dumpvars 

if [ "${arg/info}" != "$arg" ]; then
    exit 0 
fi  

unset GEOM                     # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 
export OPTICKS_MAX_RECORD=10   # change from default of 0, see sysrap/SEventConfig.cc

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

    export FOLD=$OPTICKS_OUT_FOLD
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py  
fi 

dumpvars 
exit 0 
