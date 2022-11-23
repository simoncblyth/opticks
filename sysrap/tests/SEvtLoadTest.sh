#!/bin/bash -l 

export SEvt=INFO
export SEventConfig=INFO


## these configure the directory from which to load
export GEOM=hamaLogicalPMT 
export SOpticksResource_ExecutableName=U4PMTFastSimTest 

## both these are needed to induce SEvt::Load rather than SEvt::Create
export OPTICKS_RUNNING_MODE=SRM_G4STATE_RERUN
export OPTICKS_G4STATE_RERUN=0     ## value must be > -1 and less than the number of g4state


SEvtLoadTest 



