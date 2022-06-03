#!/bin/bash -l 
usage(){ cat << EOU
U4RecorderTest.sh 
====================


EOU
}
msg="=== $BASH_SOURCE :"
logdir=/tmp/$USER/opticks/U4RecorderTest
mkdir -p $logdir 
cd $logdir 

#export DsG4Scintillation_verboseLevel=3
#export G4Cerenkov_verboseLevel=3

export DsG4Scintillation_opticksMode=3  # 3:0b11 collect gensteps and do Geant4 generation loop too 

export G4Cerenkov_DISABLE=1
#export DsG4Scintillation_DISABLE=1

#export BP=DsG4Scintillation::PostStepDoIt
#lldb__ U4RecorderTest


U4RecorderTest


echo $msg logdir $logdir


