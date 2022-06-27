#!/bin/bash -l 
usage(){ cat << EOU
cxs_raindrop.sh : CXRaindropTest combining CFBASE_LOCAL raindrop geometry with standard CFBASE basis geometry  
=================================================================================================================

Run on remote::

    cx
    ./cxs_raindrop.sh 
    PIDX=0 ./cxs_raindrop.sh     # debug print index control via envvar 

Grab from remote to laptop::

    cx
    ./cxs_raindrop.sh grab    ## currently just grabbing outputs in tmp dirs  
    ## TODO: move outputs into CSGFoundry dirs and grab geometry together with outputs  

Analysis of outputs::

    cx
    ./cxs_raindrop.sh ana 

Photon step record OpenGL animation::

    cd ~/opticks/examples/UseGeometryShader
    ./go.sh 

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
        #export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/$(SCVDLabel)/$(CSGOptiXVersion)
    else
        echo $msg run or ana mode handles Linux generated results grabbed from remote 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
        #export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/SCVD0/70000
    fi 
else
    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
    #export OPTICKS_OUT_FOLD=$CFBASE_LOCAL/$bin/$(SCVDLabel)/$(CSGOptiXVersion)
fi 

export FOLD=$CFBASE_LOCAL   ## grab fails when add the bin dir here 


export SEvt=INFO
export QEvent=INFO
export QSim=INFO


path=/tmp/storch_test/out/$(uname)/ph.npy
#path=RandomSpherical10_f8.npy


if [ -n "$path" ]; then 
    export OPTICKS_INPUT_PHOTON=$path
    if [ "${path:0:1}" == "/" -o "${path:0:1}" == "$" ]; then 
        abspath=$path
    else
        abspath=$HOME/.opticks/InputPhotons/$path
    fi
    if [ ! -f "$abspath" ]; then 
        echo $msg path $path abspath $abspath DOES NOT EXIST : create with ana/input_photons.sh OR sysrap/tests/storch_test.sh 
        exit 1 
    else
        echo $msg path $path abspath $abspath exists 
    fi 
fi 


vars="arg bin GEOM CFBASE_LOCAL OPTICKS_OUT_FOLD FOLD"
dumpvars(){ for var in $vars ; do printf "%25s : %s \n" $var ${!var} ; done ; }
dumpvars 

if [ "${arg/info}" != "$arg" ]; then
    exit 0 
fi  

unset GEOM                     # MUST unset GEOM for CSGFoundry::Load_ to load OPTICKS_KEY basis geometry 


if [ "${arg/run}" != "$arg" -o "${arg/dru}" != "$arg" -o "$arg" == "dbg" ]; then 
    logdir=/tmp/$USER/opticks/$bin
    mkdir -p $logdir
    iwd=$PWD
    cd $logdir

    export CSGMaker_makeBoxedSphere_FACTOR=10

    if [ -n "$DEBUG" -o "$arg" == "dbg" ]; then 
        case $(uname) in
           Darwin) lldb__  $bin  ;;
           Linux)  gdb $bin ;;
        esac 
    else
        $bin
    fi 

    [ $? -ne 0 ] && echo $msg RUN ERROR && exit 1 
    echo $msg logdir $logdir 
    cd $iwd
fi 

if [ "${arg/ana}" != "$arg" -o "${arg/dan}" != "$arg" ]; then 

    ${IPYTHON:-ipython} --pdb -i tests/$bin.py  
fi 

if [ "$arg" == "grab" ]; then
    EXECUTABLE=$bin  source tmpgrab.sh grab
fi 


dumpvars 
exit 0 
