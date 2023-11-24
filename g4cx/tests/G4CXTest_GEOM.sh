#!/bin/bash -l 
usage(){ cat << EOU
G4CXTest_GEOM.sh : Standalone bi-simulation with G4CXApp::Main and current GEOM 
===================================================================================

Workstation::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
    LOG=1 BP=C4CustomART::doIt ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg  

Laptop::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh grab 
    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh ana

storch::generate is used for both GPU and CPU generation
---------------------------------------------------------

* On GPU the generation is invoked at highest level CSGOptiX7.cu:simulate
* On CPU for example the stack is below, using MOCK_CURAND::

  G4RunManager::BeamOn 
  ... 
  G4RunManager::GenerateEvent 
  G4CXApp::GeneratePrimaries
  U4VPrimaryGenerator::GeneratePrimaries
  SGenerate::GeneratePhotons
  storch::generate    

EOU
}

cd $(dirname $BASH_SOURCE)
SDIR=$(pwd)

bin=G4CXTest
script=$SDIR/G4CXTest_GEOM.py 

source $HOME/.opticks/GEOM/GEOM.sh   # set GEOM and associated envvars for finding geometry

[ -n "$CVD" ] && export CUDA_VISIBLE_DEVICES=$CVD



## OPTICKS_INTEGRATION_MODE configures GPU and/or CPU optical simulation
#oim=1  # GPU only 
#oim=2  # CPU only 
oim=3   # GPU and CPU 
export OPTICKS_INTEGRATION_MODE=$oim 

## OPTICKS_EVENT_MODE configures the SEvt components to gather and save
#mode=Minimal
#mode=HitOnly
mode=StandardFullDebug
export OPTICKS_EVENT_MODE=$mode   
export OPTICKS_MAX_BOUNCE=31
export OPTICKS_NUM_EVENT=3


#num=1000
#num=5000
num=100000
NUM=${NUM:-$num}

export OPTICKS_MAX_PHOTON=100000  


srm=SRM_TORCH
#srm=SRM_INPHO
#srm=SRM_GUN
export OPTICKS_RUNNING_MODE=$srm

echo $BASH_SOURCE OPTICKS_RUNNING_MODE $OPTICKS_RUNNING_MODE

if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then 
    export SEvent_MakeGensteps_num_ph=$NUM
    #src="rectangle"
    #src="disc"
    src="sphere"

    if [ "$src" == "rectangle" ]; then
        export storch_FillGenstep_pos=0,0,0
        export storch_FillGenstep_type=rectangle
        export storch_FillGenstep_zenith=-20,20
        export storch_FillGenstep_azimuth=-20,20
    elif [ "$src" == "disc" ]; then
        export storch_FillGenstep_type=disc
        export storch_FillGenstep_radius=50      
        export storch_FillGenstep_zenith=0,1       # radial range scale
        export storch_FillGenstep_azimuth=0,1      # phi segment twopi fraction 
        export storch_FillGenstep_mom=1,0,0
        export storch_FillGenstep_pos=-80,0,0
    elif [ "$src" == "sphere" ]; then
        export storch_FillGenstep_type=sphere
        export storch_FillGenstep_radius=100    # +ve for outwards    
        export storch_FillGenstep_pos=0,0,0
        export storch_FillGenstep_distance=1.00 # frac_twopi control of polarization phase(tangent direction)
    fi 

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_INPHO" ]; then 
    echo -n 
elif [ "$OPTICKS_RUNNING_MODE" == "SRM_GUN" ]; then 
    echo -n 
fi 


TMP=${TMP:-/tmp/$USER/opticks}
export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export VERSION=0                       # used in the SEvt output directory 
export AFOLD=$BINBASE/ALL$VERSION/p001 
export BFOLD=$BINBASE/ALL$VERSION/n001 
#export BFOLD=$TMP/GEOM/$GEOM/CSGOptiXSMTest/ALL/p001  ## TMP OVERRIDE COMPARE A-WITH-A from CSGOptiXSMTest

mkdir -p $BINBASE
cd $BINBASE        ## logfile written in invoking directory 


#export G4CXOpticks__SaveGeometry_DIR=$BASE  ## optionally save geom into BASE for debug 
export G4CXApp__SensDet=PMTSDMgr             ## used for post GDML SensDet hookup


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

   #export U4GDML__VERBOSE=1 
   #export SPMTAccessor__VERBOSE=1
   export SEvt__LIFECYCLE=1  ## sparse SEvt debug output, works well alone  
}
[ -n "$LOG" ] && logging

defarg="info_run_ana"
#defarg="info_dbg_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR GEOM ${GEOM}_CFBaseFromGEOM ${GEOM}_GDMLPath VERSION TMP BASE BINBASE AFOLD BFOLD CVD CUDA_VISIBLE_DEVICES script" 


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
    source $SDIR/../../bin/rsync.sh $BINBASE
    #source $SDIR/../../bin/rsync.sh $BASE       ## widen to BASE to include the debug geometry save 
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3 
fi

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script 
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error with script $script && exit 4
fi 

exit 0 

