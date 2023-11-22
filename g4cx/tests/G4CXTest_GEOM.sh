#!/bin/bash -l 
usage(){ cat << EOU
G4CXTest_GEOM.sh : Standalone bi-simulation with G4CXApp::Main and current GEOM 
===================================================================================

Certain special GEOM strings such as "RaindropRockAirWater" 
are recognized by U4VolumeMaker::PVS_  which is called by U4VolumeMaker::PV

For the configuratuin of the raindrop see U4VolumeMaker::RaindropRockAirWater_Configure

Currently this uses the default torch genstep for the initial photons, 
see storch::FillGenstep for how to customize that. 

::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh ana

    LOG=1 BP=C4CustomART::doIt ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg  



EOU
}

cd $(dirname $BASH_SOURCE)
SDIR=$(pwd)

bin=G4CXTest
script=$SDIR/G4CXTest_GEOM.py 

source $HOME/.opticks/GEOM/GEOM.sh   # set GEOM and associated envvars for finding geometry

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

#mode=Minimal
#mode=HitOnly
mode=StandardFullDebug
export OPTICKS_EVENT_MODE=$mode   # configure what to gather and save

TMP=${TMP:-/tmp/$USER/opticks}
export BASE=$TMP/GEOM/$GEOM
export EVTBASE=$BASE/$bin
export VERSION=0                       # used in the SEvt output directory 
export AFOLD=$EVTBASE/ALL$VERSION/p001 
export BFOLD=$EVTBASE/ALL$VERSION/n001 
export G4CXOpticks__SaveGeometry_DIR=$BASE  # save geom into BASE for debug 
export G4CXApp__SensDet=PMTSDMgr 
#export U4GDML__VERBOSE=1 
#export SPMTAccessor__VERBOSE=1


logging()
{
   export SSim=INFO
   export QSim=INFO
   export QPMT=INFO
   #export SEvt=INFO
   #export U4Recorder=INFO
   #export U4StepPoint=INFO
   export U4Physics=INFO
   #export CSGFoundry=INFO
   #export CSGTarget=INFO
}
[ -n "$LOG" ] && logging

defarg="info_run_ana"
#defarg="info_dbg_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR GEOM ${GEOM}_CFBaseFromGEOM ${GEOM}_GDMLPath VERSION TMP BASE AFOLD BFOLD CVD CUDA_VISIBLE_DEVICES script" 

mkdir -p $BASE
cd $BASE


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%50s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/run}" != "$arg" ]; then
    rm -f $bin.log
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    mkdir -p $BASE
    cd $BASE
    dbg__ $(which $bin) 
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    #source $SDIR/../../bin/rsync.sh $EVTBASE
    source $SDIR/../../bin/rsync.sh $BASE       ## widen to BASE to include the debug geometry save 
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error with script $script && exit 4
fi 

exit 0 

