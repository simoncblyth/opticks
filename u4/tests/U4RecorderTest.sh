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

foldbase=/tmp/$USER/opticks/U4RecorderTest


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


#export SEvt=INFO
#export U4Random=INFO

export U4Random_flat_debug=1  ## without this all stack tags are zero 

#mode=gun
#mode=torch
mode=iphoton

export U4RecorderTest__PRIMARY_MODE=$mode
export U4VolumeMaker_RaindropRockAirWater_FACTOR=10

export ShimG4OpAbsorption_FLOAT=1 
export ShimG4OpRayleigh_FLOAT=1 


# cf U4Physics::Desc
physdesc=""
[ -n "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_FLOAT" 
[ -z "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_ORIGINAL" 
physdesc="${physdesc}_"
[ -n "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_FLOAT"
[ -z "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_ORIGINAL"

export FOLD=$foldbase/$physdesc

echo $msg physdesc $physdesc
echo $msg FOLD $FOLD


if [ "$U4RecorderTest__PRIMARY_MODE" == "iphoton" ]; then
    #path=RandomSpherical10_f8.npy
    path=/tmp/storch_test/out/$(uname)/ph.npy
    export OPTICKS_INPUT_PHOTON=$path
    echo $msg OPTICKS_INPUT_PHOTON $OPTICKS_INPUT_PHOTON
fi 



source ./IDPath_override.sh   ## non-standard IDPath to allow U4Material::LoadOri to find material properties 

#geom=BoxOfScintillator
geom=RaindropRockAirWater
export GEOM=${GEOM:-$geom}

# Note that OPTICKS_RANDOM_SEQPATH uses single quotes to prevent expansion of the '$PrecookedDir' 
# which is an SPath internal variable. Defining OPTICKS_RANDOM_SEQPATH is necessary to work with 
# more than 100k photons as the default only loads a single 100k precooked random file whereas 
# this will load ten of them allowing aligned running with up to 1M photons.
# export OPTICKS_RANDOM_SEQPATH='$PrecookedDir/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000'  


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

if [ "${arg/ab}" != "${arg}" ]; then 
    cd $srcdir 
    pwd
    ./U4RecorderTest_ab.sh
fi 


