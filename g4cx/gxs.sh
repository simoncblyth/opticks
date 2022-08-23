#!/bin/bash -l 
usage(){ cat << EOU
gxs.sh : G4CXSimulateTest : Opticks CX GPU simulation starting from Geant4 geometry auto-translated to CSG
=============================================================================================================

::

    cd ~/opticks/g4cx   # gx
    ./gxs.sh 
    ./gxs.sh info
    ./gxs.sh run
    ./gxs.sh dbg
    ./gxs.sh ana
    ./gxs.sh grab
    ./gxs.sh ab


G4CXSimulateTest invokes "SEvt::save()" writing .npy to '$DefaultOutputDir/ALL' eg::

   /tmp/$USER/opticks/$GEOM/SProc::ExecutableName/ALL
   /tmp/blyth/opticks/RaindropRockAirWater/G4CXSimulateTest/ALL 

Also the CSGFoundry geometry is written to '$DefaultOutputDir/CSGFoundry' eg:: 

   /tmp/$USER/opticks/$GEOM/SProc::ExecutableName/CSGFoundry
   /tmp/blyth/opticks/RaindropRockAirWater/G4CXSimulateTest/CSGFoundry

This assumes CFBASE is not defined. When CFBASE is defined the
geometry is written to "$CFBASE/CSGFoundry" 
HMM: but when CFBASE is defined the geometry would have been loaded from there, 
hence there would be no save done ? See G4CXOpticks::setGeometry.

HMM: maybe better to distinguish loading and saving CFBASE as a form 
of control of when to save ?


grablog/analog timings showing log lines with delta time more than 2% of total
--------------------------------------------------------------------------------

* initial CUDA access latency stands at 39% so no point working on 
  other init bottlenecks until that latency can be reduced (which might need driver update)

* HMM: a frame of pixels that is not being used may be being allocated 

::

    In [1]: log[2]
    Out[1]: 
                     timestamp :        DTS-prev :        DFS-frst :msg
    2022-08-23 23:46:28.359000 :      0.3140[39] :      0.3140[39] : INFO  [57470] [main@33] ] cu first 
    2022-08-23 23:46:28.571000 :      0.1770[22] :      0.5260[65] : INFO  [57470] [QSim::UploadComponents@111] ] new QRng 
    2022-08-23 23:46:28.590000 :      0.0180[ 2] :      0.5450[67] : INFO  [57470] [QSim::UploadComponents@128] QBnd src NP  dtype <f4(45, 4, 2, 761, 4, ) size 1095840 uifc f ebyte 4 shape.size 5 data.size 4383360 meta.size 69 names.size 45 tex QTex width 761 height 360 texObj 1 meta 0x3069a00 d_meta 0x7f3e9dc01000 tex 0x3069990
    2022-08-23 23:46:28.672000 :      0.0720[ 9] :      0.6270[78] : INFO  [57470] [CSGOptiX::initCtx@322] ]
    2022-08-23 23:46:28.696000 :      0.0230[ 3] :      0.6510[81] : INFO  [57470] [CSGOptiX::initPIP@333] ]
    2022-08-23 23:46:28.805000 :      0.0350[ 4] :      0.7600[94] : INFO  [57470] [CSGFoundry::getFrame@2880]  fr sframe::desc inst 0 frs -1
    2022-08-23 23:46:28.839000 :      0.0200[ 2] :      0.7940[98] : INFO  [57470] [CSGOptiX::launch@794]  (width, height, depth) ( 1920,1080,1) 0.0201


                             - :                 :                 :G4CXSimulateTest.log
    2022-08-23 23:46:28.045000 :                 :                 :start
    2022-08-23 23:46:28.853000 :                 :                 :end
                             - :                 :      0.8080[100] :total seconds
                             - :                 :      2.0000[100] :pc_cut


Using single GPU reduces first contact latency a little, total time down to 0.7410 seconds.


EOU
}


defarg="run"
arg=${1:-$defarg}

case $arg in 
  fold) QUIET=1 ;; 
esac


bin=G4CXSimulateTest
log=$bin.log
gxsdir=$(dirname $BASH_SOURCE)
source $gxsdir/../bin/COMMON.sh 


BASE=$GEOMDIR/$bin
UBASE=${BASE//$HOME\/}    # UBASE is BASE relative to HOME to handle rsync between different HOME
FOLD=$BASE/ALL            # corresponds SEvt::save() with SEvt::SetReldir("ALL")

notes(){ cat << EON

* When BASE is not within $HOME eg its in /tmp then UBASE and BASE are the same.  

EON
}


# NB FOLD is not used by run, but it is used by ana

if [ "info" == "$arg" ]; then 
    vars="BASH_SOURCE gxsdir GEOM GEOMDIR CFBASE BASE UBASE FOLD OPTICKS_INPUT_PHOTON"
    for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "fold" == "$arg" ]; then 
    echo $FOLD 
fi 




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
}
loglevels

#export U4VolumeMaker_PVG_WriteNames=1


if [ "run" == "$arg" ]; then 
    echo $msg run $bin log $log 
    if [ -f "$log" ]; then 
       rm $log 
    fi 

    $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE run $bin error && exit 1 
fi 

if [ "dbg" == "$arg" ]; then 
    case $(uname) in
        Linux) gdb_ $bin -ex r  ;;
        Darwin) lldb__ $bin ;; 
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg $bin error && exit 2 
fi


if [ "grablog" == "$arg" ]; then
    scp P:opticks/g4cx/$log .
fi 

if [ "analog" == "$arg" ]; then 
    echo $msg analog log $log 
    if [ -f "$log" ]; then 
        LOG=$log $gxsdir/../bin/log.sh 
    fi 
fi 


if [ "ana" == "$arg" ]; then 
    export FOLD 
    export CFBASE=$BASE/CSGFoundry
    ${IPYTHON:-ipython} --pdb -i $gxsdir/tests/$bin.py     
    [ $? -ne 0 ] && echo $BASH_SOURCE ana $bin error && exit 3 
fi 

if [ "grab" == "$arg" ]; then 
    source $gxsdir/../bin/rsync.sh $UBASE 
fi 

if [ "ab" == "$arg" ]; then
    cd $gxsdir
    ./gxs_ab.sh 
fi 

exit 0 

