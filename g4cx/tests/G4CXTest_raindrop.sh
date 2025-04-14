#!/bin/bash
usage(){ cat << EOU
G4CXTest_raindrop.sh : Standalone bi-simulation with G4CXApp::Main
===================================================================

Certain special GEOM strings such as "RaindropRockAirWater" 
are recognized by U4VolumeMaker::PVS_  which is called by U4VolumeMaker::PV

For the configuratuin of the raindrop see U4VolumeMaker::RaindropRockAirWater_Configure

Currently this uses the default torch genstep for the initial photons, 
see storch::FillGenstep for how to customize that. 

::

    ~/opticks/g4cx/tests/G4CXTest_raindrop.sh
    ~/opticks/g4cx/tests/G4CXTest_raindrop_CPU.sh

    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_raindrop.sh ana

    PICK=A MODE=3 SELECT="TO BT BR BR BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 
    PICK=B MODE=3 SELECT="TO BT BR BR SA"          ~/opticks/g4cx/tests/G4CXTest_raindrop.sh ana

    NUM=1000000 PICK=B MODE=3 SELECT="TO BT BR BR BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
vars="" 
DIR=$(pwd)
bin=G4CXTest
script=G4CXTest_raindrop.py 

export GEOM=RaindropRockAirWater  # GEOM is identifier for a geometry 

vars="$vars BASH_SOURCE PWD DIR bin script GEOM"

# THESE ARE NOW THE DEFAULTS 
export U4VolumeMaker_RaindropRockAirWater_RINDEX=0,0,1,1.333
export U4VolumeMaker_RaindropRockAirWater_MATS=VACUUM,G4_Pb,G4_AIR,G4_WATER
export U4VolumeMaker_RaindropRockAirWater_HALFSIDE=100
export U4VolumeMaker_RaindropRockAirWater_DROPSHAPE=Box  # default:Orb  (Box also impl) 

if [ -n "$KLUDGE" ]; then
    export U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE=1 
fi

if [ "$U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE" == "1" ]; then
    version=1 
else
    version=0 
fi 

export VERSION=$version  # used in the SEvt output directory 
vars="$vars VERSION"


#export Local_DsG4Scintillation_DISABLE=1
export G4CXOpticks__SaveGeometry_DIR=$HOME/.opticks/GEOM/$GEOM


#num=1000
#num=5000
num=H1
NUM=${NUM:-$num}

## For torch running MUST NOW configure the below two NUM envvars 
## with the same number of comma delimited values, or just 1 value

export OPTICKS_NUM_PHOTON=$NUM  
export OPTICKS_NUM_GENSTEP=1
export OPTICKS_RUNNING_MODE="SRM_TORCH"
export OPTICKS_MAX_SLOT=M1


vars="$vars OPTICKS_NUM_PHOTON OPTICKS_NUM_GENSTEP OPTICKS_RUNNING_MODE"


if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then  
    #export SEvent_MakeGenstep_num_ph=$NUM   ## NO LONGER USED ? 

    #src="rectangle"
    #src="disc"
    src="circle_inwards_hemi"

    if [ "$src" == "rectangle" ]; then
        export storch_FillGenstep_pos=0,0,0
        export storch_FillGenstep_type=rectangle
        export storch_FillGenstep_zenith=-20,20
        export storch_FillGenstep_azimuth=-20,20
    elif [ "$src" == "disc" ]; then
        export storch_FillGenstep_type=disc
        export storch_FillGenstep_radius=50        # radius
        export storch_FillGenstep_zenith=0,1       # radial range scale
        export storch_FillGenstep_azimuth=0,1      # phi segment twopi fraction 
        export storch_FillGenstep_mom=1,0,0
        export storch_FillGenstep_pos=-80,0,0
    elif [ "$src" == "circle_inwards_hemi" ]; then
        export storch_FillGenstep_type=circle
        export storch_FillGenstep_radius=-50    # -ve means inwards
        export storch_FillGenstep_pos=0,0,50
        export storch_FillGenstep_azimuth=0.5,1
    fi 
    vars="$vars src"
fi


#oim=2  # CPU only 
oim=3  # GPU and CPU optical simulation
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-$oim}

#mode=Minimal
mode=DebugLite
export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-$mode} # configure what to gather and save


# below directories match those used by SEvt saving 
# in order to be able to load SEvt into python analysis script
# 
export TMP=${TMP:-/tmp/$USER/opticks} 
evtfold=$TMP/GEOM/$GEOM

export AFOLD=$evtfold/$bin/ALL$VERSION/A000
export BFOLD=$evtfold/$bin/ALL$VERSION/B000


vars="$vars OPTICKS_INTEGRATION_MODE OPTICKS_EVENT_MODE TMP evtfold AFOLD BFOLD"


event_debug()
{
    export SEventConfig=INFO
    export SEvt__LIFECYCLE=1
    export SEvt__MINIMAL=1
    export SEvt=INFO
    export SEvent=INFO

    type $FUNCNAME 
}
[ -n "$EVENT_DEBUG" ] && event_debug

logging()
{
   export U4Recorder=INFO
   export U4StepPoint=INFO
   export U4Physics=INFO
   #export CSGFoundry=INFO
   #export CSGTarget=INFO
}
[ -n "$LOG" ] && logging

defarg="info_run_ana"
#defarg="info_dbg_ana"
arg=${1:-$defarg}

[ -n "$BP" ] && echo $BASH_SOURCE : override arg to info_dbg as BP $BP is defined && arg=info_dbg 

if [ -n "$BP" ]; then 
   DEBUG_GENIDX=10000 
   export U4VPrimaryGenerator__GeneratePrimaries_From_Photons_DEBUG_GENIDX=$DEBUG_GENIDX
   # for DEBUG_GENIDX > -1 will only generate one photon : for debugging purposes 
   echo $BASH_SOURCE :  DEBUG_GENIDX $DEBUG_GENIDX OPTICKS_NUM_PHOTON $OPTICKS_NUM_PHOTON
fi 


vars="$vars CUDA_VISIBLE_DEVICES BP defarg arg" 



if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%30s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    source $DIR/../../bin/dbg__.sh 
    dbg__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    source $DIR/../../bin/rsync.sh $evtfold
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then
    export SAVE_SEL=1

    if command -v ${IPYTHON:-ipython} &> /dev/null 
    then
        ${IPYTHON:-ipython} --pdb -i $script 
    else
        echo $BASH_SOURCE - IPYTHON NOT AVAILABLE - TRY PYTHON 
        ${PYTHON:-python} -i $script 
    fi 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error with script $script && exit 4
fi 

exit 0 

