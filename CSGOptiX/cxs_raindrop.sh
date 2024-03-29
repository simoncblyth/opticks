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

Changing the geometry::

    c
    ./CSGMakerTest.sh 


For A-B comparison with U4RecorderTest. Manually arrange same geometry and then::

    u4t
    ./U4RecorderTest.sh run
    ./U4RecorderTest.sh ab    # after grabbing the corresponding cxs_rainbow.sh outputs


EOU
}

export GEOM=BoxedSphere

cvd=0
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}  
export CVDLabel="CVD${CUDA_VISIBLE_DEVICES}" 


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
    else
        echo $msg run or ana mode handles Linux generated results grabbed from remote 
        export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
    fi 
else
    export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/$GEOM
fi 

export FOLD=$CFBASE_LOCAL   ## grab fails when add the bin dir here 


loglevels()
{
    export Dummy=INFO
    #export SEvt=INFO
    #export QEvent=INFO
    #export QSim=INFO
}
loglevels




source ../bin/OPTICKS_INPUT_PHOTON.sh


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

    ## Useless setting this here, must be set when run c:CSGMakerTest.sh 
    ##export CSGMaker_makeBoxedSphere_FACTOR=10

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

if [ "$arg" == "grab" -o "$arg" == "graby" ]; then
    EXECUTABLE=$bin  source tmpgrab.sh $arg
fi 


dumpvars 
exit 0 
