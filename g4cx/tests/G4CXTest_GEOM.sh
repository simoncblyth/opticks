#!/bin/bash  
usage(){ cat << EOU
G4CXTest_GEOM.sh : Standalone optical only bi-simulation with G4CXApp::Main and current GEOM 
================================================================================================

Standalone optical Geant4 initialization is faster than embedded Geant4 + Opticks but still ~120s to voxelize. 


Usage examples
----------------

Typically on workstation::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh
    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg
    LOG=1 BP=C4CustomART::doIt ~/opticks/g4cx/tests/G4CXTest_GEOM.sh dbg  

    PRECOOKED=1 ~/o/G4CXTest_GEOM.sh

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh info
    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh chi2


Following inclusion of _${TEST} in LOGDIR can compare different tests easily 
-----------------------------------------------------------------------------

::

    TEST=jan20 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh chi2

    TEST=feb20 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh chi2  ## small diffs only following DEBUG_TAG switch off from different randoms  
    TEST=ref1  ~/opticks/g4cx/tests/G4CXTest_GEOM.sh chi2  ## same as feb20



Reporting on the nature of the geometry conversion
------------------------------------------------------

Q: This is running from GDML, is the force triangulation done ? 
A: probably not without the config::

   export stree__force_triangulate_solid='filepath:$HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt'


TODO: confirmation in metadata of what is triangulated in the geometry, stree level for details and CSGFoundry::descSolidIntent
for the collective situation 


The info would be persisted with the stree info, but no persist ? Can be reported by stree::desc_lvid::

     318     const char*      force_triangulate_solid ;
     319     std::vector<int> force_triangulate_lvid ;




Laptop::

    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh grab 
    EYE=0,-400,0 ~/opticks/g4cx/tests/G4CXTest_GEOM.sh ana
    ~/opticks/g4cx/tests/G4CXTest_GEOM.sh pvcap 


Where possible its better to use pure Opticks simulation (no bi-simulation) 
booting from a persisted geometry during testing due to the ~2s initialization time, eg with::

    ~/opticks/CSGOptiX/cxs_min.sh


chi2 subcommand uses a faster C++ sseq index comparison implementation of the python/NumPy chi2 calc
---------------------------------------------------------------------------------------------------------

Uses::

   $SDIR/../../sysrap/tests/sseq_index_test.sh info_run_ana

*  ~/opticks/sysrap/tests/sseq_index_test.sh



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
## not the normal cd to the SDIR, as need to cd to LOGDIR but use scripts from SDIR

bin=G4CXTest
ana_script=$SDIR/G4CXTest_GEOM.py 
dna_script=$SDIR/G4CXTest.py 

source $HOME/.opticks/GEOM/GEOM.sh   # set GEOM and associated envvars for finding geometry
export ${GEOM}_GDMLPathFromGEOM=$HOME/.opticks/GEOM/$GEOM/origin.gdml
export stree__force_triangulate_solid='filepath:$HOME/.opticks/GEOM/${GEOM}_meshname_stree__force_triangulate_solid.txt'
export SSim__stree_level=1 

## CURRENTLY CAN MANUALLY CHECK WHAT IS TRIANGULATED BY UPPING THE LOGGING 
## TODO: some reporting in metadata of which solids are triangulated that gets saved with event metadata


#test=small
#test=ref1
test=ref5
#test=large_scan
TEST=${TEST:-$test}

if [ "$TEST" == "ref1" ]; then 

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=M1
   opticks_max_slot=M1

elif [ "$TEST" == "ref5" ]; then 

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=M5
   opticks_max_slot=M5

elif [ "$TEST" == "small" ]; then 

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=H1
   opticks_max_slot=M1

elif [ "$TEST" == "tiny_scan" ]; then 

   opticks_num_event=10
   opticks_num_genstep=1x10
   opticks_num_photon=K1:10
   opticks_max_slot=M1

elif [ "$TEST" == "large_scan" ]; then 

   opticks_num_event=20
   opticks_num_genstep=1x10,10x10
   opticks_num_photon=H1:10,M2,3,5,7,10,20,40,60,80,100
   opticks_max_slot=M100  

elif [ "$TEST" == "large_evt" ]; then 

   opticks_num_event=1
   opticks_num_genstep=40
   opticks_num_photon=M180   ## M200 gives OOM with TITAN RTX 24G with debug arrays enabled
   opticks_max_slot=M180

fi 



version=98
VERSION=${VERSION:-$version}
export VERSION    ## used in SEvt output directory name ALL$VERSION


ctx=$(TEST=ContextString sbuild_test)  ## eg Debug_Philox see sbuild.h 
#export OPTICKS_EVENT_NAME=$ctx  # used by SEventConfig::EventReldir "OPTICKS_EVENT_RELDIR"
export OPTICKS_EVENT_NAME=${ctx}_${TEST}


opticks_event_reldir=ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none}   
export OPTICKS_EVENT_RELDIR='ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none}'  ## this is the default in SEventConfig
## above two lines match SEventConfig::_DefaultEventReldir
## resolution of first line happens here for bash consumption, resolution of 2nd done in SEventConfig
## HMM: MAYBE SIMPLIFY BY JUST DEFINING RELDIR HERE, BUT THE CODE DEFAULT IS HANDY ?

TMP=${TMP:-/tmp/$USER/opticks}

export SCRIPT=$(basename $BASH_SOURCE)
export BASE=$TMP/GEOM/$GEOM
export LOGDIR=$BASE/$bin/$opticks_event_reldir
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




opticks_start_index=0
opticks_max_bounce=31
opticks_integration_mode=3

#opticks_running_mode=SRM_DEFAULT
opticks_running_mode=SRM_TORCH
#opticks_running_mode=SRM_INPUT_PHOTON
#opticks_running_mode=SRM_INPUT_GENSTEP    ## NOT IMPLEMENTED FOR GEANT4
#opticks_running_mode=SRM_GUN


export OPTICKS_NUM_EVENT=${OPTICKS_NUM_EVENT:-$opticks_num_event}  
export OPTICKS_NUM_GENSTEP=${OPTICKS_NUM_GENSTEP:-$opticks_num_genstep}   
export OPTICKS_NUM_PHOTON=${OPTICKS_NUM_PHOTON:-$opticks_num_photon}  

export OPTICKS_RUNNING_MODE=${OPTICKS_RUNNING_MODE:-$opticks_running_mode}   # SRM_TORCH/SRM_INPUT_PHOTON/SRM_INPUT_GENSTEP
export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-$opticks_event_mode}         # what arrays are saved eg Hit,HitPhoton,HitPhotonSeq 

 
export OPTICKS_MAX_SLOT=${OPTICKS_MAX_SLOT:-$opticks_max_slot}  
export OPTICKS_START_INDEX=${OPTICKS_START_INDEX:-$opticks_start_index}
export OPTICKS_MAX_BOUNCE=${OPTICKS_MAX_BOUNCE:-$opticks_max_bounce}
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-$opticks_integration_mode}


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

    ## UNTESTED
    export OPTICKS_INPUT_PHOTON=RainXZ_Z230_10k_f8.npy
    export OPTICKS_INPUT_PHOTON_FRAME=NNVT:0:1000

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

allarg="info_env_run_dbg_report_plot_grab_gevt_chi2_ana_dna_mpcap_mppub_pvcap_pvpub"
defarg="info_env_run_report_ana"
#defarg="info_dbg_ana"
[ -n "$BP" ] && defarg="info_dbg" 

arg=${1:-$defarg}


gdb__() 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}


vars="BASH_SOURCE allarg defarg arg SDIR GEOM ${GEOM}_CFBaseFromGEOM ${GEOM}_GDMLPathFromGEOM bin VERSION"
vars="$vars TMP BASE PWD"
vars="$vars OPTICKS_EVENT_NAME OPTICKS_EVENT_RELDIR"
vars="$vars LOGDIR AFOLD BFOLD CUDA_VISIBLE_DEVICES ana_script dna_script TEST" 


if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%50s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/env}" != "$arg" ]; then 
    env | grep OPTICKS | perl -n -e 'm/(\S*)=(\S*)/ && printf("%50s : %s\n", $1, $2) ' -
fi 

if [ "${arg/run}" != "$arg" ]; then
    rm -f $bin.log
    $bin
    pwd
    ls -alst $bin.log
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    gdb__ $bin
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
    echo $FUNCNAME - grab - WARNING - debug events can be VERY LARGE - use gevt to rsync single event  
    source $SDIR/../../bin/rsync.sh $LOGDIR   
    [ $? -ne 0 ] && echo $BASH_SOURCE grab error && exit 3 
fi

if [ "${arg/gevt}" != "$arg" ]; then
    source $SDIR/../../bin/rsync.sh $AFOLD
    source $SDIR/../../bin/rsync.sh $BFOLD
fi 

if [ "${arg/chi2}" != "$arg" ]; then
    $SDIR/../../sysrap/tests/sseq_index_test.sh info_run_ana
    [ $? -ne 0 ] && echo $BASH_SOURCE sseq_index_test chi2 ERROR && exit 3 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $ana_script 
    [ $? -ne 0 ] && echo $BASH_SOURCE pdb error with ana_script $ana_script && exit 4
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${PYTHON:-python} $ana_script 
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

