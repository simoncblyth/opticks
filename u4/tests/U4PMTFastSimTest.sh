#!/bin/bash -l 
usage(){ cat << EOU
U4PMTFastSimTest.sh  : TODO needs new name 
=================================================

::

    N=0 ./U4PMTFastSimTest.sh   # old geom  
    N=1 ./U4PMTFastSimTest.sh   # new natural geom

    PID=726 ./U4PMTFastSimTest.sh nana


    N=1 MODE=0 ./U4PMTFastSimTest.sh ph  # no GUI with NumPy 
    N=1 MODE=2 ./U4PMTFastSimTest.sh ph  # 2D GUI with matplotlib
    N=1 MODE=3 ./U4PMTFastSimTest.sh ph  # 3D GUI with pyvista


EOU
}

bin=U4PMTFastSimTest
export GEOM=hamaLogicalPMT
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/$bin



## process DISABLE/ENABLE controlling u4/tests/U4Physics.cc U4Physics::ConstructOp
export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
#export G4OpAbsorption_DISABLE=1
#export G4OpRayleigh_DISABLE=1
#export G4OpBoundaryProcess_DISABLE=1
export G4FastSimulationManagerProcess_ENABLE=1  
## HMM: should FastSim process be switched off for N=1 running ? 

export U4RecorderTest__PRIMARY_MODE=torch  # hmm seems iphoton and torch do same thing internally 

## u4/tests/U4PMTFastSimTest.cc
export BeamOn=${BeamOn:-1}


geomscript=$GEOM.sh 
version=${N:-0}

if [ -f "$geomscript" ]; then  
    source $geomscript $version
else
    echo $BASH_SOURCE : no geomscript $geomscript
fi 

# python ana level presentation 
export LOC=skip





log=${bin}.log
logN=${bin}_${version}.log


running_mode=SRM_G4STATE_SAVE  
#running_mode=SRM_G4STATE_RERUN

case $running_mode in 
   SRM_G4STATE_SAVE)  reldir=ALL ;; 
   SRM_G4STATE_RERUN) reldir=SEL$version ;; 
esac


if [ "$running_mode" == "SRM_G4STATE_RERUN" -a "$version" == "1" ]; then

   ## when using natural geometry need to apply some burns to
   ## jump ahead in a way that corresponds to the consumption 
   ## for navigating the fake volumes in the old complex geomerty 

   ./UU_BURN.sh 
   export SEvt__UU_BURN=/tmp/UU_BURN.npy
fi 


## sysrap/SEventConfig 
export OPTICKS_RUNNING_MODE=$running_mode   # see SEventConfig::RunningMode
export OPTICKS_MAX_BOUNCE=20                # can go upto 31 now that extended sseq.h 
export OPTICKS_G4STATE_RERUN=726
export OPTICKS_EVENT_MODE=StandardFullDebug


#num_ph=2
#num_ph=10
num_ph=1000
#num_ph=50000

radius=250
#radius=0
[ $num_ph -lt 11  ] && radius=0

export SEvent_MakeGensteps_num_ph=$num_ph
#export storch_FillGenstep_type=disc
export storch_FillGenstep_type=line     
export storch_FillGenstep_radius=$radius

# up from line below equator
#export storch_FillGenstep_pos=0,0,-20
#export storch_FillGenstep_mom=0,0,1

# down from line outside Pyrex
export storch_FillGenstep_pos=0,0,200
export storch_FillGenstep_mom=0,0,-1


loglevel(){
   export U4Recorder=INFO
   export U4Physics=INFO
   export junoPMTOpticalModel=INFO
   export junoPMTOpticalModelSimple=INFO
   #export SEvt=INFO
   export SEventConfig=INFO
   export InstrumentedG4OpBoundaryProcess=INFO
   export ShimG4OpAbsorption=INFO
   export ShimG4OpRayleigh=INFO
}

if [ "$running_mode" == "SRM_G4STATE_RERUN" ]; then 
   loglevel  ## switch on logging when doing single photon RERUN
fi 


defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 

    [ -f "$log" ] && echo $BASH_SOURCE rename log $log to logN $logN && mv $log $logN    
    ## HMM: probably an envvar can change the logname directly ? 
fi 

if [ "$arg" == "dbg" ]; then
    #export BP=MixMaxRng::flat
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "$arg" == "ana" -o "$arg" == "nana" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nana" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 

if [ "$arg" == "cf" -o "$arg" == "ncf" ]; then
    [ "$arg" == "ncf" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_cf.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE cf error && exit 4
fi 

if [ "$arg" == "ph" -o "$arg" == "nph" ]; then
    export FOLD=$BASE/$reldir
    [ "$arg" == "nph" ] && export MODE=0
    ${IPYTHON:-ipython} --pdb -i ${bin}_ph.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ph error && exit 5
fi 

exit 0 

