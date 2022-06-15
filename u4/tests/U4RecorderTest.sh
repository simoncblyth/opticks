#!/bin/bash -l 
usage(){ cat << EOU
U4RecorderTest.sh 
====================

::

    cd ~/opticks/u4/tests
    ./U4RecorderTest.sh 


EOU
}
msg="=== $BASH_SOURCE :"

arg=${1:-run_ana}

srcdir=$PWD
logdir=/tmp/$USER/opticks/U4RecorderTest
mkdir -p $logdir 

export FOLD=/tmp/$USER/opticks/U4RecorderTest
#export DsG4Scintillation_verboseLevel=3
#export G4Cerenkov_verboseLevel=3
export DsG4Scintillation_opticksMode=3  # 3:0b11 collect gensteps and do Geant4 generation loop too 

#export G4Cerenkov_DISABLE=1
#export DsG4Scintillation_DISABLE=1

#pidx=0
#export PIDX=${PIDX:-$pidx}

#gidx=117
#export GIDX=${GIDX:-$gidx}

#export U4Material=INFO



#mode=gun
#mode=torch
mode=iphoton

export U4RecorderTest__PRIMARY_MODE=$mode

if [ "$U4RecorderTest__PRIMARY_MODE" == "iphoton" ]; then
    export OPTICKS_INPUT_PHOTON=RandomSpherical10_f8.npy
fi 



source ./IDPath_override.sh   ## non-standard IDPath to allow U4Material::LoadOri to find material properties 

#geom=BoxOfScintillator
geom=RaindropRockAirWater
export GEOM=${GEOM:-$geom}


if [ "${arg/run}" != "${arg}" ]; then 
    cd $logdir 
    #export BP=DsG4Scintillation::PostStepDoIt
    #lldb__ U4RecorderTest
    U4RecorderTest
    [ $? -ne 0 ] && echo $msg run error && exit 1 

    echo $msg logdir $logdir
fi 

if [ "${arg/dbg}" != "${arg}" ]; then 
    cd $logdir 
    #export BP=DsG4Scintillation::PostStepDoIt
    lldb__ U4RecorderTest
    echo $msg logdir $logdir
fi 

if [ "${arg/clean}" != "${arg}" ]; then
   cd $FOLD 
   pwd
   ls -l *.npy *.txt *.log
   read -p "$msg Enter YES to delete these : " ans
   if [ "$ans" == "YES" ] ; then 
       echo $msg proceeding
       rm *.npy *.txt *.log
   else
       echo $msg skip 
   fi 
fi 

if [ "${arg/ana}" != "${arg}" ]; then 
    cd $srcdir 
    pwd

    ${IPYTHON:-ipython} --pdb -i U4RecorderTest.py 
fi 


