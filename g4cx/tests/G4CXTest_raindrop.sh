#!/bin/bash -l 
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
    PICK=B MODE=3 SELECT="TO BT BR BR SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh ana

    NUM=1000000 PICK=B MODE=3 SELECT="TO BT BR BR BR BR BT SA" ~/opticks/g4cx/tests/G4CXTest_raindrop.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))
DIR=$(pwd)

bin=G4CXTest
script=G4CXTest_raindrop.py 

export GEOM=RaindropRockAirWater  # GEOM is identifier for a geometry 

# THESE ARE NOW THE DEFAULTS 
export U4VolumeMaker_RaindropRockAirWater_RINDEX=0,0,1,1.333
export U4VolumeMaker_RaindropRockAirWater_MATS=VACUUM,G4_Pb,G4_AIR,G4_WATER
export U4VolumeMaker_RaindropRockAirWater_HALFSIDE=100
export U4VolumeMaker_RaindropRockAirWater_DROPSHAPE=Box  # default:Orb  (Box also impl) 

#export U4Recorder__PreUserTrackingAction_Optical_UseGivenVelocity_KLUDGE=1 


export G4CXOpticks__SaveGeometry_DIR=$HOME/.opticks/GEOM/$GEOM

if [ -n "$CVD" ]; then 
    export CUDA_VISIBLE_DEVICES=$CVD
fi


#num=1000
#num=5000
num=H1
NUM=${NUM:-$num}

export OPTICKS_NUM_PHOTON=$NUM   ## supports comma delimited values

export OPTICKS_RUNNING_MODE="SRM_TORCH"

#export U4VPrimaryGenerator__GeneratePrimaries_From_Photons_DEBUG_GENIDX=50000
# for DEBUG_GENIDX > -1 will only generate one photon : for debugging purposes 


if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then  
    #export SEvent_MakeGenstep_num_ph=$NUM   ## NO LONGER USED ? 

    #src="rectangle"
    #src="disc"
    src="circle_inwards_hemi"

    ## TODO: shoot all angles from just inside drop to check after TIR speed

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
tmpbase=${TMP:-/tmp/$USER/opticks} 
evtfold=$tmpbase/GEOM/$GEOM

export VERSION=0  # used in the SEvt output directory 

# THESE ARE THE OLD FOLDR PATHS
#export AFOLD=$evtfold/$bin/ALL$VERSION/p001 
#export BFOLD=$evtfold/$bin/ALL$VERSION/n001

export AFOLD=$evtfold/$bin/ALL$VERSION/A000
export BFOLD=$evtfold/$bin/ALL$VERSION/B000



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

vars="BASH_SOURCE GEOM VERSION TMP AFOLD BFOLD evtfold CVD CUDA_VISIBLE_DEVICES BP arg" 

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
       Linux) gdb -ex r --args $bin  ;;
       Darwin) lldb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    source $DIR/../../bin/rsync.sh $evtfold
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then
    export SAVE_SEL=1
 
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error with script $script && exit 4
fi 

exit 0 

