#!/bin/bash -l 
usage(){ cat << EOU
G4CXTest_GEOM.sh : Standalone optical only bi-simulation with G4CXApp::Main and current GEOM 
================================================================================================

Standalone optical Geant4 initialization is faster than embedded Geant4 + Opticks but still ~120s to voxelize. 

Workstation::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg
    LOG=1 BP=C4CustomART::doIt ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg  


    PRECOOKED=1 ~/o/G4CXTest_GEOM.sh



Laptop::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh grab 
    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh ana

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh pvcap 



Where possible its better to use pure Opticks simulation (no bi-simulation) 
booting from a persisted geometry during testing due to the ~2s initialization time, eg with::

    ~/opticks/CSGOptiX/cxs_min.sh


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

SDIR=$(dirname $(realpath $BASH_SOURCE))

bin=G4CXTest
ana_script=$SDIR/G4CXTest_GEOM.py 
dna_script=$SDIR/G4CXTest.py 

source $HOME/.opticks/GEOM/GEOM.sh   # set GEOM and associated envvars for finding geometry

[ -n "$CVD" ] && export CUDA_VISIBLE_DEVICES=$CVD



version=98
VERSION=${VERSION:-$version}
export VERSION    ## used in SEvt output directory name ALL$VERSION


TMP=${TMP:-/tmp/$USER/opticks}
export SCRIPT=$(basename $BASH_SOURCE)
export BASE=$TMP/GEOM/$GEOM
export LOGDIR=$BASE/$bin/ALL$VERSION
export AFOLD=$LOGDIR/A000 
export BFOLD=$LOGDIR/B000 
#export BFOLD=$TMP/GEOM/$GEOM/CSGOptiXSMTest/ALL/A000  ## TMP OVERRIDE COMPARE A-WITH-A from CSGOptiXSMTest

pick=A
export PICK=${PICK:-$pick}  ## PICK is only used by ana, not running 
case $PICK in 
    A) FOLD=$AFOLD ;; 
    B) FOLD=$BFOLD ;; 
   AB) FOLD=$AFOLD ;;
esac
export FOLD    ## changed sreport to use SREPORT_FOLD to avoid clash 
export STEM=G4CXTest_GEOM_${PICK}

mkdir -p $LOGDIR
cd $LOGDIR            ## logfile written in invoking directory 


#export G4CXOpticks__SaveGeometry_DIR=$BASE  ## optionally save geom into BASE for debug 
export G4CXApp__SensDet=PMTSDMgr             ## used for post GDML SensDet hookup



knobs()
{
   type $FUNCNAME 

   local exceptionFlags
   local debugLevel
   local optLevel

   #exceptionFlags=STACK_OVERFLOW   
   exceptionFlags=NONE

   #debugLevel=DEFAULT
   debugLevel=NONE
   #debugLevel=FULL

   #optLevel=DEFAULT
   #optLevel=LEVEL_0
   optLevel=LEVEL_3

 
   #export PIP__max_trace_depth=1
   export PIP__CreatePipelineOptions_exceptionFlags=$exceptionFlags # NONE/STACK_OVERFLOW/TRACE_DEPTH/USER/DEBUG
   export PIP__CreateModule_debugLevel=$debugLevel  # DEFAULT/NONE/MINIMAL/MODERATE/FULL   (DEFAULT is MINIMAL)
   export PIP__linkPipeline_debugLevel=$debugLevel  # DEFAULT/NONE/MINIMAL/MODERATE/FULL   
   export PIP__CreateModule_optLevel=$optLevel      # DEFAULT/LEVEL_0/LEVEL_1/LEVEL_2/LEVEL_3  

   #export Ctx=INFO
   #export PIP=INFO
   #export CSGOptiX=INFO
}
knobs


case $VERSION in 
 0) opticks_event_mode=Minimal ;;
 1) opticks_event_mode=Hit ;; 
 2) opticks_event_mode=HitPhoton ;; 
 3) opticks_event_mode=HitPhoton ;;    ## USING 3 FOR LEAK TEST 
 4) opticks_event_mode=HitPhotonSeq ;; 
 5) opticks_event_mode=HitSeq ;; 
98) opticks_event_mode=DebugLite ;;
99) opticks_event_mode=DebugHeavy ;;
esac 



#test=small
test=reference
#test=large_scan
TEST=${TEST:-$test}

if [ "$TEST" == "reference" ]; then 

   opticks_num_photon=M1
   opticks_max_photon=M1
   opticks_num_event=1

elif [ "$TEST" == "small" ]; then 

   opticks_num_photon=H1
   opticks_max_photon=M1
   opticks_num_event=1


elif [ "$TEST" == "tiny_scan" ]; then 

   opticks_num_photon=K1:10
   opticks_max_photon=M1
   opticks_num_event=10

elif [ "$TEST" == "large_scan" ]; then 

   opticks_num_photon=H1:10,M2,3,5,7,10,20,40,60,80,100
   opticks_max_photon=M100   ## cost: QRng init time + VRAM 
   opticks_num_event=20

elif [ "$TEST" == "large_evt" ]; then 

   opticks_num_photon=M200   ## OOM with TITAN RTX 24G 
   opticks_max_photon=M200   ## cost: QRng init time + VRAM 
   opticks_num_event=1

fi 

opticks_start_index=0
opticks_max_bounce=31
opticks_integration_mode=3

#opticks_running_mode=SRM_DEFAULT
opticks_running_mode=SRM_TORCH
#opticks_running_mode=SRM_INPUT_PHOTON
#opticks_running_mode=SRM_INPUT_GENSTEP    ## NOT IMPLEMENTED FOR GEANT4
#opticks_running_mode=SRM_GUN

export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-$opticks_event_mode}   
export OPTICKS_NUM_PHOTON=${OPTICKS_NUM_PHOTON:-$opticks_num_photon}   
export OPTICKS_NUM_EVENT=${OPTICKS_NUM_EVENT:-$opticks_num_event}  
export OPTICKS_MAX_PHOTON=${OPTICKS_MAX_PHOTON:-$opticks_max_photon}  
export OPTICKS_START_INDEX=${OPTICKS_START_INDEX:-$opticks_start_index}
export OPTICKS_MAX_BOUNCE=${OPTICKS_MAX_BOUNCE:-$opticks_max_bounce}
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-$opticks_integration_mode}
export OPTICKS_RUNNING_MODE=${OPTICKS_RUNNING_MODE:-$opticks_running_mode}


if [ -n "$PRECOOKED" ]; then 
    ## SPECIAL SETTING TO GET G4CXApp::GeneratePrimaries
    ## TO USE THE SAME RANDOMS AS CURAND : SO CAN COMPARE 
    ## WITH THE SAME START PHOTONS
    export SGenerate__GeneratePhotons_RNG_PRECOOKED=1
    export s_seq__SeqPath_DEFAULT_LARGE=1   ## enable M1, without this are limited to K100 
fi 


if [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then 
    #export SEvent_MakeGenstep_num_ph=$NUM   ## trumped by OPTICKS_NUM_PHOTON
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

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_PHOTON" ]; then 
    echo -n 

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_GENSTEP" ]; then 
    echo -n 

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_GUN" ]; then 
    echo -n 
fi 

logging()
{
   #export SSim=INFO
   #export QSim=INFO
   #export QPMT=INFO
   #export SEvt=INFO
   #export U4Recorder=INFO
   #export U4StepPoint=INFO
   #export U4Physics=INFO
   #export CSGFoundry=INFO
   #export CSGTarget=INFO

   #export U4GDML__VERBOSE=1 
   #export SPMTAccessor__VERBOSE=1
   #export SEvt__LIFECYCLE=1  ## sparse SEvt debug output, works well alone  
  
   export G4CXOpticks=INFO

}
[ -n "$LOG" ] && logging
[ -n "$LIFECYCLE" ] && export SEvt__LIFECYCLE=1

defarg="info_env_run_report_ana"
#defarg="info_dbg_ana"
arg=${1:-$defarg}

vars="BASH_SOURCE SDIR GEOM ${GEOM}_CFBaseFromGEOM ${GEOM}_GDMLPath VERSION TMP BASE LOGDIR AFOLD BFOLD CVD CUDA_VISIBLE_DEVICES ana_script" 


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%50s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/env}" != "$arg" ]; then 
    env | grep OPTICKS | perl -n -e 'm/(\S*)=(\S*)/ && printf("%50s : %s\n", $1, $2) ' -
fi 

if [ "${arg/run}" != "$arg" ]; then
    rm -f $bin.log
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2 
fi 

if [ "${arg/report}" != "$arg" ]; then
    sreport
    [ $? -ne 0 ] && echo $BASH_SOURCE sreport error && exit 1 
fi 

if [ "${arg/plot}" != "$arg" ]; then
    runprof=1 $SDIR/../../sysrap/tests/sreport.sh ana
    [ $? -ne 0 ] && echo $BASH_SOURCE sreport.plot error && exit 1 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    source $SDIR/../../bin/rsync.sh $LOGDIR   ## COULD BE VERY LARGE : BETTER TO GRAB SINGLE EVT  
    [ $? -ne 0 ] && echo $BASH_SOURCE grab error && exit 3 
fi

if [ "${arg/gevt}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $AFOLD
    source $OPTICKS_HOME/bin/rsync.sh $BFOLD
fi 

if [ "${arg/chi2}" != "$arg" ]; then
    #sseq_index_test
    $OPTICKS_HOME/sysrap/tests/sseq_index_test.sh info_run_ana
    [ $? -ne 0 ] && echo $BASH_SOURCE sseq_index_test chi2 ERROR && exit 3 
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana_script 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error with ana_script $ana_script && exit 4
fi 

if [ "${arg/dna}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $dna_script 
    [ $? -ne 0 ] && echo $BASH_SOURCE dna error with dna_script $dna_script && exit 4
fi 





if [ "$arg" == "mpcap" -o "$arg" == "mppub" -o "$arg" == "pvcap" -o "$arg" == "pvpub" ]; then
    export CAP_BASE=$FOLD/figs
    export CAP_REL=G4CXTest_GEOM
    export CAP_STEM=$STEM
    case $arg in  
       mpcap) source mpcap.sh cap  ;;  
       pvcap) source pvcap.sh cap  ;;  
       mppub) source mpcap.sh env  ;;  
       pvpub) source pvcap.sh env  ;;  
    esac
    if [ "$arg" == "mppub" -o "$arg" == "pvpub" ]; then 
        source epub.sh 
    fi  
fi 

exit 0 

