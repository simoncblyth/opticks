#!/bin/bash -l 
usage(){ cat << EOU
uxs.sh : formerly U4RecorderTest.sh : Geant4 simulation with Opticks recording every random consumption of every step of every photon
=========================================================================================================================================

::

    cd ~/opticks/u4   # u4
    ./uxs.sh 

    BP=DsG4Scintillation::PostStepDoIt ./uxs.sh dbg 


    ./uxs.sh run
    ./uxs.sh dbg
    ./uxs.sh clean
    ./uxs.sh ana
    ./uxs.sh ab


EOU
}
msg="=== $BASH_SOURCE :"

case $(uname) in 
   Linux) defarg="run" ;;
   Darwin) defarg="run_ana" ;; 
esac
arg=${1:-$defarg}


bin=U4RecorderTest
srcdir=$PWD
logdir=/tmp/$USER/opticks/$bin
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

export U4Random_select_action=interrupt   ## dumps stack and breaks in debugger to check the process

#mode=gun
#mode=torch
mode=iphoton
export U4RecorderTest__PRIMARY_MODE=$mode

if [ "$U4RecorderTest__PRIMARY_MODE" == "iphoton" ]; then
    source ../bin/OPTICKS_INPUT_PHOTON.sh     
fi 

source ./IDPath_override.sh   
# IDPath_override.sh : non-standard IDPath to allow U4Material::LoadOri to find material properties 
# HMM probably doing nothing now that are using U4Material::LoadBnd ?

source ../bin/GEOM_.sh 


if [ -n "$CFBASE" ]; then 
    echo $msg CFBASE from GEOM_.sh : $CFBASE
else 
    CFBASE=/tmp/$USER/opticks/G4CXSimulateTest/$GEOM
fi



#sel=PIDX_0_
sel=ALL


export ShimG4OpAbsorption_FLOAT=1 
export ShimG4OpRayleigh_FLOAT=1 

# cf U4Physics::Desc
physdesc=""
[ -n "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_FLOAT" 
[ -z "$ShimG4OpAbsorption_FLOAT" ] && physdesc="${physdesc}ShimG4OpAbsorption_ORIGINAL" 
physdesc="${physdesc}_"
[ -n "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_FLOAT"
[ -z "$ShimG4OpRayleigh_FLOAT" ]   && physdesc="${physdesc}ShimG4OpRayleigh_ORIGINAL"


export FOLD=$foldbase/$physdesc/$GEOM/$sel




## CFBASE in different tree because it is kinda foreign 
## HMM: maybe should copy it ?



if [ "${arg/info}" != "${arg}" ]; then 
    vars="GEOM FOLD foldbase physdesc sel" 
    for var in $vars ; do printf " %30s : %s \n" $var ${!var}  ; done 
fi 


if [ -d "${CFBASE}/CSGFoundry" ]; then 
    export CFBASE
    echo $msg cfbase/CSGFoundry dir exists so exporting CFBASE $CFBASE
else
    echo $msg cfbase/CSGFoundry dir does not exist : NOT EXPORTING CFBASE
    exit 1
fi 





# Note that OPTICKS_RANDOM_SEQPATH uses single quotes to prevent expansion of the '$PrecookedDir' 
# which is an SPath internal variable. Defining OPTICKS_RANDOM_SEQPATH is necessary to work with 
# more than 100k photons as the default only loads a single 100k precooked random file whereas 
# this will load ten of them allowing aligned running with up to 1M photons.
# export OPTICKS_RANDOM_SEQPATH='$PrecookedDir/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000'  




loglevels()
{
    export Dummy=INFO
    #export U4Material=INFO
    #export SEvt=INFO
    #export U4Random=INFO
}
loglevels




if [ "${arg/run}" != "${arg}" ]; then 
    cd $logdir 
    $bin 
    [ $? -ne 0 ] && echo $msg run error && exit 1 

    echo $msg logdir $logdir
fi 

if [ "${arg/dbg}" != "${arg}" ]; then 
    cd $logdir 
    case $(uname) in 
       Linux)  gdb_ $bin;;
       Darwin) lldb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $msg dbg error && exit 2 
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
    ${IPYTHON:-ipython} --pdb -i tests/$bin.py 
fi 

if [ "${arg/grab}" != "${arg}" ]; then 
    echo $msg grab FOLD $FOLD
    source ../bin/rsync.sh $FOLD
fi 

if [ "${arg}" == "ab" ]; then 
    cd $srcdir 
    pwd
    #fold_mode=TMP
    #fold_mode=KEEP
    #fold_mode=LOGF
    fold_mode=GEOM
    export FOLD_MODE=${FOLD_MODE:-$fold_mode}
    source ../bin/AB_FOLD.sh 
    ${IPYTHON:-ipython} --pdb -i tests/${bin}_ab.py $*  
fi 




