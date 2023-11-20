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

    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_raindrop.sh ana

EOU
}

cd $(dirname $BASH_SOURCE)
DIR=$(pwd)

bin=G4CXTest
script=G4CXTest_raindrop.py 

export GEOM=RaindropRockAirWater  # GEOM is identifier for a geometry 

# THESE ARE NOW THE DEFAULTS 
export U4VolumeMaker_RaindropRockAirWater_RINDEX=0,0,1,1.333
export U4VolumeMaker_RaindropRockAirWater_MATS=VACUUM,G4_Pb,G4_AIR,G4_WATER
export U4VolumeMaker_RaindropRockAirWater_HALFSIDE=100


export G4CXOpticks__SaveGeometry_DIR=$HOME/.opticks/GEOM/$GEOM

if [ -n "$CVD" ]; then 
    export CUDA_VISIBLE_DEVICES=$CVD
fi


num=1000
#num=5000
#num=1000000
NUM=${NUM:-$num}

export SEvent_MakeGensteps_num_ph=$NUM

#src="rectangle"
src="disc"

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
fi 


#oim=2  # CPU only 
oim=3  # GPU and CPU optical simulation
export OPTICKS_INTEGRATION_MODE=$oim 

mode=Minimal
#mode=StandardFullDebug
export OPTICKS_EVENT_MODE=$mode   # configure what to gather and save

# below directories match those used by SEvt saving 
# in order to be able to load SEvt into python analysis script
# 
tmpbase=${TMP:-/tmp/$USER/opticks} 
evtfold=$tmpbase/GEOM/$GEOM

export VERSION=0  # used in the SEvt output directory 
export AFOLD=$evtfold/$bin/ALL$VERSION/p001 
export BFOLD=$evtfold/$bin/ALL$VERSION/n001 


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

vars="BASH_SOURCE GEOM VERSION TMP AFOLD BFOLD evtfold CVD CUDA_VISIBLE_DEVICES" 

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
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error with script $script && exit 4
fi 

exit 0 

