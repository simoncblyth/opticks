#!/bin/bash -l 
usage(){ cat << EOU
gxr.sh : G4CXRenderTest 
=============================================================================================================

::

    cd ~/opticks/g4cx   # gx
    ./gxr.sh 
    ./gxr.sh run
    ./gxr.sh dbg
    ./gxr.sh grab
    ./gxr.sh grablog
    ./gxr.sh analog


analog delta times more than 2% of total
--------------------------------------------------

See bin/log.py for logfile analysis with time filtering 

EOU
}

msg="=== $BASH_SOURCE :"

case $(uname) in 
  Linux)  defarg="run"  ;;
  Darwin) defarg="ls"  ;;
esac

arg=${1:-$defarg}

case $arg in
  fold) QUIET=1 ;;
  analog)  QUIET=1 ;;
esac




bin=G4CXRenderTest
log=$bin.log
gxrdir=$(dirname $BASH_SOURCE)

source $gxrdir/../bin/COMMON.sh 

## bin/COMMON sources bin/GEOM_.sh and bin/OPTICKS_INPUT_PHOTON.sh
## BUT the content of those is very user specific so have moved
## geometry to $HOME/.opticks/GEOM/GEOM.sh 
## THIS KINDA THING IS IN USERLAND BUT COULD SUGGEST 
## INPUT PHOTONS ALSO CONFIGURED IN THE SAME GEOM.sh FILE
## OR FILES THAT THAT FILE SOURCES 


eye=-0.4,0,0
moi=-1
export EYE=${EYE:-$eye} 
export MOI=${MOI:-$moi}


# HMM could do loglevels in COMMON.sh ?
# NO : WHAT LOGGING TO SWITCH ON DEPENDS ON EACH SCRIPT
# SO IT SHOULD BE THERE

loglevels()
{
    export Dummy=INFO
    #export SGeoConfig=INFO
    export SEventConfig=INFO
    export SEvt=INFO          # lots of AddGenstep output, messing with timings
    #export Ctx=INFO
    export QSim=INFO
    export QBase=INFO
    export SSim=INFO
    export SBT=INFO
    export IAS_Builder=INFO
    #export QEvent=INFO 
    export CSGOptiX=INFO
    export G4CXOpticks=INFO 
    export CSGFoundry=INFO
    #export GInstancer=INFO
    #export X4PhysicalVolume=INFO
    #export U4VolumeMaker=INFO
    export Frame=INFO
}
loglevels



if [ "run" == "$arg" ]; then 
    echo $msg run $bin log $log 
    if [ -f "$log" ]; then 
       rm $log 
    fi 

    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "dbg" == "$arg" ]; then 
    case $(uname) in
        Linux) gdb $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi


if [ "grablog" == "$arg" ]; then
    scp P:opticks/g4cx/$log .
fi 

if [ "analog" == "$arg" ]; then 
    echo $msg analog log $log 
    if [ -f "$log" ]; then 
        LOG=$log $gxrdir/../bin/log.sh 
    fi 
fi 


# FOLD is not an input to C++ running, but it is used by the below : ls ana grab jpg  
export FOLD=/tmp/$USER/opticks/$GEOM/$bin
name=cx$MOI.jpg
path=$FOLD/$name

if [ "ls" == "$arg" ]; then 
   echo $msg FOLD $FOLD 
   echo $msg date $(date)
   ls -alst $FOLD
fi 

if [ "ana" == "$arg" ]; then 
    export CFBASE=$FOLD
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3 
fi 

if [ "grab" == "$arg" ]; then 
    source ../bin/rsync.sh $FOLD
    open $path 
fi 

if [ "jpg" == "$arg" ]; then 
    mkdir -p $(dirname $path)
    scp P:$path $path 
    open $path 
fi 

exit 0 

