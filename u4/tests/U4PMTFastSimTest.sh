#!/bin/bash -l 
usage(){ cat << EOU
U4PMTFastSimTest.sh
======================

::

    PID=726 ./U4PMTFastSimTest.sh nana

EOU
}

## process DISABLE/ENABLE controlling u4/tests/U4Physics.cc U4Physics::ConstructOp
export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
export G4FastSimulationManagerProcess_ENABLE=1  

#running_mode=SRM_G4STATE_SAVE  
running_mode=SRM_G4STATE_RERUN
export OPTICKS_RUNNING_MODE=$running_mode   # see SEventConfig::RunningMode
export OPTICKS_MAX_BOUNCE=20                # can go upto 31 now that extended sseq.h 

case $running_mode in 
   SRM_G4STATE_SAVE)  reldir=ALL ;; 
   SRM_G4STATE_RERUN) reldir=SEL ;; 
esac

export OPTICKS_G4STATE_RERUN=726
export OPTICKS_EVENT_MODE=StandardFullDebug

export InstrumentedG4OpBoundaryProcess=INFO


export GEOM=hamaLogicalPMT
export U4RecorderTest__PRIMARY_MODE=torch 
# hmm seems iphoton and torch do same thing internally 
export BeamOn=${BeamOn:-1}

export hama_FastCoverMaterial=Cheese    ## declProp config of Manager
export hama_UsePMTOpticalModel=1
export hama_UseNaturalGeometry=1

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


bin=U4PMTFastSimTest
log=$bin.log

loglevel(){
   export U4Recorder=INFO
   export junoPMTOpticalModel=INFO
   export junoPMTOpticalModelSimple=INFO
   #export SEvt=INFO
   export SEventConfig=INFO
}


if [ "$running_mode" == "SRM_G4STATE_RERUN" ]; then 
   loglevel
fi 


defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then
    [ -f "$log" ] && rm $log 
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

if [ "$arg" == "ana" -o "$arg" == "nana" ]; then
    #export FOLD=/tmp/SFastSim_Debug
    export FOLD=/tmp/$USER/opticks/GEOM/$GEOM/$bin/$reldir

    [ "$arg" == "nana" ] && export NOGUI=1
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 



exit 0 

