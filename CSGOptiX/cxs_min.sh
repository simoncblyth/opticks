#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Usage::

    ~/o/cxs_min.sh
    ~/o/cxs_min.sh info
    ~/o/cxs_min.sh run       ## create SEvt 
    ~/o/cxs_min.sh report    ## summarize SEvt metadata   

Debug::

    BP=SEvt::SEvt               ~/opticks/CSGOptiX/cxs_min.sh
    BP=SEvent::MakeTorchGenstep ~/opticks/CSGOptiX/cxs_min.sh

Analysis/Plotting::

    ~/o/cxs_min.sh grab 
    EVT=A000 ~/o/cxs_min.sh ana

    MODE=2 SEL=1 ~/o/cxs_min.sh ana 
    EVT=A005     ~/o/cxs_min.sh ana 
    EVT=A010     ~/o/cxs_min.sh ana

    PLOT=scatter MODE=3 ~/o/cxs_min.sh pvcap 

Monitor for GPU memory leaks::

    ~/o/sysrap/smonitor.sh build_run  # start monitor

    TEST=large_scan ~/o/cxs_min.sh   

    # CTRL-C smonitor.sh session sending SIGINT to process which saves smonitor.npy

    ~/o/sysrap/smonitor.sh grab  ## back to laptop
    ~/o/sysrap/smonitor.sh ana   ## plot 
    

EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

case $(uname) in
   Linux) defarg=run_report_info ;;
   Darwin) defarg=ana ;;
esac

[ -n "$BP" ] && defarg=dbg
[ -n "$PLOT" ] && defarg=ana

arg=${1:-$defarg}


bin=CSGOptiXSMTest
script=$SDIR/cxs_min.py

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

export EVT=${EVT:-A000}
export BASE=${TMP:-/tmp/$USER/opticks}/GEOM/$GEOM
export BINBASE=$BASE/$bin
export SCRIPT=$(basename $BASH_SOURCE)


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


version=1
VERSION=${VERSION:-$version}   ## see below currently using VERSION TO SELECT OPTICKS_EVENT_MODE
export VERSION
## VERSION CHANGES OUTPUT DIRECTORIES : SO USEFUL TO ARRANGE SEPARATE STUDIES

export LOGDIR=$BINBASE/ALL$VERSION
export AFOLD=$BINBASE/ALL$VERSION/$EVT
export STEM=ALL${VERSION}_${PLOT}

#export BFOLD=$BASE/G4CXTest/ALL0/$EVT  ## comparison with "A" from another executable
#export BFOLD=$BASE/jok-tds/ALL0/A001    ## comparison with "A" from another executable

mkdir -p $LOGDIR 
cd $LOGDIR 
LOGFILE=$bin.log

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}

case $VERSION in 
 0) opticks_event_mode=Minimal ;;
 1) opticks_event_mode=Hit ;; 
 2) opticks_event_mode=HitPhoton ;; 
 3) opticks_event_mode=HitPhoton ;; 
 4) opticks_event_mode=HitPhotonSeq ;; 
 5) opticks_event_mode=HitSeq ;; 
98) opticks_event_mode=DebugLite ;;
99) opticks_event_mode=DebugHeavy ;;  # formerly StandardFullDebug
esac 

test=reference
TEST=${TEST:-$test}

if [ "$TEST" == "reference" ]; then 

   opticks_num_photon=M1
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
opticks_integration_mode=1

#opticks_running_mode=SRM_DEFAULT
opticks_running_mode=SRM_TORCH
#opticks_running_mode=SRM_INPUT_PHOTON
#opticks_running_mode=SRM_INPUT_GENSTEP
#opticks_running_mode=SRM_GUN

export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-$opticks_event_mode}
export OPTICKS_NUM_PHOTON=${OPTICKS_NUM_PHOTON:-$opticks_num_photon} 
export OPTICKS_NUM_EVENT=${OPTICKS_NUM_EVENT:-$opticks_num_event}
export OPTICKS_MAX_PHOTON=${OPTICKS_MAX_PHOTON:-$opticks_max_photon}
export OPTICKS_START_INDEX=${OPTICKS_START_INDEX:-$opticks_start_index}
export OPTICKS_MAX_BOUNCE=${OPTICKS_MAX_BOUNCE:-$opticks_max_bounce}
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-$opticks_integration_mode}
export OPTICKS_RUNNING_MODE=${OPTICKS_RUNNING_MODE:-$opticks_running_mode}



if [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_GENSTEP" ]; then 

    igs=$BASE/jok-tds/ALL0/p001/genstep.npy 
    ##igs=$BASE/jok-tds/ALL0   # TODO: impl handling a sequence of input genstep 
    export OPTICKS_INPUT_GENSTEP=$igs
    [ ! -f "$igs" ] && echo $BASH_SOURCE : FATAL : NO SUCH PATH : igs $igs && exit 1

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_PHOTON" ]; then 

    #ipho=RainXZ_Z195_1000_f8.npy      ## ok 
    #ipho=RainXZ_Z230_1000_f8.npy      ## ok
    #ipho=RainXZ_Z230_10k_f8.npy       ## ok
    ipho=RainXZ_Z230_100k_f8.npy
    #ipho=RainXZ_Z230_X700_10k_f8.npy  ## X700 to illuminate multiple PMTs
    #ipho=GridXY_X700_Z230_10k_f8.npy 
    #ipho=GridXY_X1000_Z1000_40k_f8.npy

    #moi=-1
    #moi=sWorld:0:0
    #moi=NNVT:0:0
    #moi=NNVT:0:50
    moi=NNVT:0:1000
    #moi=PMT_20inch_veto:0:1000
    #moi=sChimneyAcrylic 

    # SEventConfig
    export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$ipho};
    export OPTICKS_INPUT_PHOTON_FRAME=${MOI:-$moi}

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then 

    #export SEvent_MakeGenstep_num_ph=100000  NOT USED WHEN USING OPTICKS_NUM_PHOTON
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

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_GUN" ]; then 

    echo -n 

fi 



logging(){ 
    export CSGFoundry=INFO
    export CSGOptiX=INFO
    export QEvent=INFO 
    export QSim=INFO
    export SEvt__LIFECYCLE=1
}
[ -n "$LOG" ] && logging
[ -n "$LIFECYCLE" ] && export SEvt__LIFECYCLE=1


vars="GEOM LOGDIR BINBASE CVD CUDA_VISIBLE_DEVICES SDIR FOLD LOG NEVT"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
fi 

if [ "${arg/env}" != "$arg" ]; then 
    env | grep OPTICKS | perl -n -e 'm/(\S*)=(\S*)/ && printf("%50s : %s\n", $1, $2) ' -
fi 

if [ "${arg/fold}" != "$arg" ]; then
    echo $FOLD
fi 

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   if [ -f "$LOGFILE" ]; then 
       echo $BASH_SOURCE : run : delete prior LOGFILE $LOGFILE 
       rm "$LOGFILE" 
   fi 

   if [ "${arg/run}" != "$arg" ]; then
       date +"%Y-%m-%d %H:%M:%S.%3N  %N : [$BASH_SOURCE "
       $bin
       [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
       date +"%Y-%m-%d %H:%M:%S.%3N  %N : ]$BASH_SOURCE "
   elif [ "${arg/dbg}" != "$arg" ]; then
       dbg__ $bin 
       [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1 
   fi 
fi 

if [ "${arg/meta}" != "$arg" ]; then
   if [ -f "run_meta.txt" -a -n "$OPTICKS_SCAN_INDEX"  -a -d "$OPTICKS_SCAN_INDEX" ] ; then
       cp run_meta.txt $OPTICKS_SCAN_INDEX/run_meta.txt
   fi 
   [ $? -ne 0 ] && echo $BASH_SOURCE meta error && exit 1 
fi 


if [ "${arg/report}" != "$arg" ]; then
   sreport
   [ $? -ne 0 ] && echo $BASH_SOURCE sreport error && exit 1 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $LOGDIR
fi 

if [ "${arg/gevt}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $LOGDIR/$EVT
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi 

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$AFOLD/figs
    export CAP_REL=cxs_min
    export CAP_STEM=$STEM
    case $arg in  
       pvcap) source pvcap.sh cap  ;;  
       mpcap) source mpcap.sh cap  ;;  
       pvpub) source pvcap.sh env  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 

