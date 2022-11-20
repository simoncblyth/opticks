#!/bin/bash -l 

export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
export G4FastSimulationManagerProcess_ENABLE=1

loglevel(){
   export U4Recorder=INFO
   export junoPMTOpticalModel=INFO
}
#loglevel

export GEOM=hamaLogicalPMT
export U4RecorderTest__PRIMARY_MODE=torch 
# hmm seems iphoton and torch do same thing internally 
export BeamOn=1


export SEvent_MakeGensteps_num_ph=50000
#export SEvent_MakeGensteps_num_ph=1000
export storch_FillGenstep_type=line     # disc
export storch_FillGenstep_radius=250

# up from line below equator
#export storch_FillGenstep_pos=0,0,-20
#export storch_FillGenstep_mom=0,0,1

# down from line outside Pyrex
export storch_FillGenstep_pos=0,0,200
export storch_FillGenstep_mom=0,0,-1





bin=U4PMTFastSimTest

defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 


if [ "$arg" == "ana" -o "$arg" == "nana" ]; then
    export FOLD=/tmp/SFastSim_Debug
    [ "$arg" == "nana" ] && export NOGUI=1
    ${IPYTHON:-ipython} --pdb -i $bin.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 3
fi 


exit 0 

