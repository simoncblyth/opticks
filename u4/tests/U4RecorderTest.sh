#!/bin/bash -l 
usage(){ cat << EOU
U4RecorderTest.sh : Geant4 simulation with Opticks recording every random consumption of every step of every photon
=====================================================================================================================

::

    cd ~/opticks/u4/tests  # u4t 
    ./U4RecorderTest.sh 

    BP=DsG4Scintillation::PostStepDoIt ./U4RecorderTest.sh dbg 


    ./U4RecorderTest.sh run
    ./U4RecorderTest.sh dbg
    ./U4RecorderTest.sh clean
    ./U4RecorderTest.sh ana
    ./U4RecorderTest.sh ab


EOU
}
msg="=== $BASH_SOURCE :"

arg=${1:-run_ana}

srcdir=$PWD
logdir=/tmp/$USER/opticks/U4RecorderTest
mkdir -p $logdir 
foldbase=$logdir


export DsG4Scintillation_opticksMode=3  # 3:0b11 collect gensteps and do Geant4 generation loop too 

#export DsG4Scintillation_verboseLevel=3
#export DsG4Scintillation_DISABLE=1
#export G4Cerenkov_verboseLevel=3
#export G4Cerenkov_DISABLE=1
#pidx=0
#gidx=117
#export PIDX=${PIDX:-$pidx}
#export GIDX=${GIDX:-$gidx}

loglevels()
{
    export Dummy=INFO
    #export U4Material=INFO
    #export SEvt=INFO
    #export U4Random=INFO
}
loglevels


export U4Random_flat_debug=1  ## without this all stack tags are zero 

#mode=gun
#mode=torch
mode=iphoton
export U4RecorderTest__PRIMARY_MODE=$mode

if [ "$U4RecorderTest__PRIMARY_MODE" == "iphoton" ]; then
    source ../../bin/OPTICKS_INPUT_PHOTON.sh     
fi 

source ./IDPath_override.sh   
# IDPath_override.sh : non-standard IDPath to allow U4Material::LoadOri to find material properties 
# HMM probably doing nothing now that are using U4Material::LoadBnd ?

source ../../bin/GEOM_.sh 



export ShimG4OpAbsorption_FLOAT=1 
export ShimG4OpRayleigh_FLOAT=1 

pidx=0
export ShimG4OpAbsorption_PIDX=$pidx
export ShimG4OpRayleigh_PIDX=$pidx

# cf U4Physics::Desc
physdesc=""
[ -n "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_FLOAT" 
[ -z "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_ORIGINAL" 
physdesc="${physdesc}_"
[ -n "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_FLOAT"
[ -z "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_ORIGINAL"


export FOLD=$foldbase/$physdesc/$GEOM
echo $msg physdesc $physdesc
echo $msg GEOM $GEOM
echo $msg FOLD $FOLD


cfbase=/tmp/$USER/opticks/G4CXSimulateTest/$GEOM
if [ -d "${cfbase}/CSGFoundry" ]; then 
    export CFBASE=$cfbase
    echo $msg cfbase/CSGFoundry dir exists so defined CFBASE $CFBASE
fi 


# Note that OPTICKS_RANDOM_SEQPATH uses single quotes to prevent expansion of the '$PrecookedDir' 
# which is an SPath internal variable. Defining OPTICKS_RANDOM_SEQPATH is necessary to work with 
# more than 100k photons as the default only loads a single 100k precooked random file whereas 
# this will load ten of them allowing aligned running with up to 1M photons.
# export OPTICKS_RANDOM_SEQPATH='$PrecookedDir/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000'  


if [ "${arg/run}" != "${arg}" ]; then 
    cd $logdir 
    U4RecorderTest
    [ $? -ne 0 ] && echo $msg run error && exit 1 

    echo $msg logdir $logdir
fi 

if [ "${arg/dbg}" != "${arg}" ]; then 
    cd $logdir 
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

