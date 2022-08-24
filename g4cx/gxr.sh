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

::

    In [1]: log[2]
    Out[1]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 23:46:36.137000 :      0.2620[42] :      0.2620[42] : INFO  [57512] [main@24] ] cu first 
    2022-08-23 23:46:36.240000 :      0.0740[12] :      0.3650[59] : INFO  [57512] [CSGOptiX::initCtx@322] ]
    2022-08-23 23:46:36.264000 :      0.0230[ 4] :      0.3890[63] : INFO  [57512] [CSGOptiX::initPIP@333] ]
    2022-08-23 23:46:36.299000 :      0.0130[ 2] :      0.4240[68] : INFO  [57512] [IAS_Builder::CollectInstances@77]  i   25601 gasIdx   2 sbtOffset   3094 gasIdx_sbtOffset.size   3
    2022-08-23 23:46:36.374000 :      0.0350[ 6] :      0.4990[80] : INFO  [57512] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 23:46:36.390000 :      0.0130[ 2] :      0.5150[83] : INFO  [57512] [CSGOptiX::launch@794]  (width, height, depth) ( 1920,1080,1) 0.0126
    2022-08-23 23:46:36.495000 :      0.0960[15] :      0.6200[100] : INFO  [57512] [Frame::snap@155] ] writeJPG 


                             - :                 :                 :G4CXRenderTest.log
    2022-08-23 23:46:35.875000 :                 :                 :start
    2022-08-23 23:46:36.496000 :                 :                 :end
                             - :                 :      0.6210[100] :total seconds
                             - :                 :      2.0000[100] :pc_cut

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

eye=-0.4,0,0
moi=-1
export EYE=${EYE:-$eye} 
export MOI=${MOI:-$moi}


# HMM could do loglevels in COMMON.sh ?

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

