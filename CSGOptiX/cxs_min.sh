#!/bin/bash
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Uses ~oneline main::

     CSGOptiX::SimulateMain();

Usage::

    ~/o/cxs_min.sh
    ~/o/cxs_min.sh info
    ~/o/cxs_min.sh run       ## create SEvt
    ~/o/cxs_min.sh report    ## summarize SEvt metadata



This script is used for many purposes in development and testing
making it require understanding and often editing before use,
plus the setting of the TEST envvar to select the type of
test to perform.

This script runs the CSGOptiXSMTest executable which has no Geant4 dependency,
so it is restricted to purely optical running and loads the persisted CSGFoundry
from ~/.opticks/GEOM/$GEOM/CSGFoundry using GEOM envvar
set by ~/.opticks/GEOM/GEOM.sh

This script is most commonly used for "torch" running where initial photons
are generated in simple patterns and with numbers of photons configured by the
script. Input photons and input gensteps can also be configured. Small scans
simulating multiple events with varying numbers of photons can also be configured,
often simply by selecting a TEST envvar value.


Examples::

    TEST=vvlarge_evt ~/o/cxs_min.sh
         ## caution tries to simulate a billion photons
         ## for JUNO writes ~12GB of hits



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


pdb1
   ipython cxs_min.py

pdb0
   as pdb1 with MODE=0 for no pyvista or matplotlib usage
   useful for loading/examining the SEvt with ipython
   from anywhere

EOU
}

vars=""
SDIR=$(dirname $(realpath $BASH_SOURCE))

vars="$vars BASH_SOURCE SDIR"

case $(uname) in
   Linux) defarg=run_report_info ;;
   Darwin) defarg=ana ;;
esac

[ -n "$BP" ] && defarg=dbg
[ -n "$PLOT" ] && defarg=ana

arg=${1:-$defarg}
allarg=info_env_fold_run_dbg_meta_report_grab_grep_gevt_du_pdb1_pdb0_AB_ana_pvcap_pvpub_mpcap_mppub
vars="$vars defarg arg allarg"


bin=CSGOptiXSMTest
script=$SDIR/cxs_min.py
script_AB=$SDIR/cxs_min_AB.py
vars="$vars bin script script_AB"


External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM
if [ -n "$GEOM" -a -n "${!External_CFBaseFromGEOM}" -a -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/CSGFoundry/prim.npy" ]; then
    echo $BASH_SOURCE - External GEOM setup detected
    vv="External_CFBaseFromGEOM ${External_CFBaseFromGEOM}"
    for v in $vv ; do printf "%40s : %s \n" "$v" "${!v}" ; done
else
    ## development source tree usage : where need to often switch between geometries
    source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar, use GEOM bash function to setup/edit
    #export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
fi

vars="$vars GEOM"

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
export EVT=${EVT:-A000}
export BASE=$TMP/GEOM/$GEOM
export BINBASE=$BASE/$bin
export SCRIPT=$(basename $BASH_SOURCE)

vars="$vars TMP EVT BASE BINBASE SCRIPT"

vars="$vars ${GEOM}_CFBaseFromGEOM ${GEOM}_GDMLPathFromGEOM"

# pulled out complicated Resolve_CFBaseFromGEOM.sh from here
# instead rely on user to setup geometry access envvar
# in GEOM.sh or elsewhere

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


   #export NPFold__substamp_DUMP=1


}


#version=0
version=1
#version=98   ## set to 98 for low stats debugging

export VERSION=${VERSION:-$version}   ## see below currently using VERSION TO SELECT OPTICKS_EVENT_MODE
## VERSION CHANGES OUTPUT DIRECTORIES : SO USEFUL TO ARRANGE SEPARATE STUDIES



vars="$vars version VERSION"


#test=debug
#test=ref1
#test=ref5
#test=ref8
#test=ref10
#test=ref10_multilaunch
#test=input_genstep
#test=input_genstep_muon

#test=input_photon_chimney
#test=input_photon_nnvt
#test=input_photon_target
#test=input_photon_wp_pmt
#test=input_photon_wp_pmt_side
#test=input_photon_wp_pmt_semi
#test=input_photon_s_pmt
#test=input_photon_poolcover
#test=input_photon_poolcover_refine

#test=large_evt
#test=vlarge_evt
#test=vvlarge_evt
#test=vvvlarge_evt
#test=vvvvlarge_evt
test=vvvvvlarge_evt
#test=vvvvvvlarge_evt

#test=medium_scan

export TEST=${TEST:-$test}


#ctx=Debug_XORWOW
#ctx=Debug_Philox

case $(uname) in
   Darwin) ctx=Debug_Philox ;;
    Linux) ctx=$(TEST=ContextString sbuild_test) ;;
esac

export OPTICKS_EVENT_NAME=${ctx}_${TEST}
## SEventConfig::Initialize_EventName asserts OPTICKS_EVENT_NAME sbuild::Matches config of the build


opticks_event_reldir=ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none}   ## matches SEventConfig::_DefaultEventReldir OPTICKS_EVENT_RELDIR
export OPTICKS_EVENT_RELDIR='ALL${VERSION:-0}_${OPTICKS_EVENT_NAME:-none}'  ## this is the default in SEventConfig
# opticks_event_reldir is resolved here, OPTICKS_EVENT_RELDIR resolved by SEvt/SEventConfig

vars="$vars test TEST opticks_event_reldir OPTICKS_EVENT_RELDIR"

case $TEST in
ref10_multilaunch) alt_TEST=ref10_onelaunch ;;
ref10_onelaunch)   alt_TEST=ref10_multilaunch ;;
esac

alt_opticks_event_reldir=ALL${VERSION:-0}_${alt_TEST}
vars="$vars alt_TEST alt_opticks_event_reldir"



export LOGDIR=$BINBASE/$opticks_event_reldir
export AFOLD=$BINBASE/$opticks_event_reldir/$EVT
export STEM=${opticks_event_reldir}_${PLOT}

#export BFOLD=$BASE/G4CXTest/ALL0/$EVT
#export BFOLD=$BASE/jok-tds/ALL0/A001
#BFOLD_NOTE="comparison with A from another executable"

#export BFOLD=$BINBASE/$alt_opticks_event_reldir/$EVT   # comparison with alt_TEST
#BFOLD_NOTE="comparison with alt_TEST:$alt_TEST"
BFOLD_NOTE="defining BFOLD makes python script do SAB comparison"


mkdir -p $LOGDIR
cd $LOGDIR
export SProf__WRITE=1  ## enable SProf::Write of SProf.txt into LOGDIR


LOGFILE=$bin.log

vars="$vars LOGDIR AFOLD BFOLD BFOLD_NOTE STEM LOGFILE"


case $VERSION in
 0) opticks_event_mode=Minimal ;;
 1) opticks_event_mode=Hit ;;
 2) opticks_event_mode=HitPhoton ;;
 3) opticks_event_mode=HitPhoton ;;
 4) opticks_event_mode=HitPhotonSeq ;;
 5) opticks_event_mode=HitSeq ;;
98) opticks_event_mode=DebugLite ;;
99) opticks_event_mode=DebugHeavy ;;
esac

# WIP: decouple VERSION from opticks_event_mode, better for event_mode to be controlled via TEST
# demote use of VERSION to special checks with a separate output folder

vars="$vars opticks_event_mode"


if [ "$TEST" == "debug" ]; then

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=100
   opticks_running_mode=SRM_TORCH
   #opticks_max_photon=M1

elif [ "$TEST" == "ref1" ]; then

   opticks_num_event=1
   opticks_num_genstep=10
   opticks_num_photon=M1
   opticks_running_mode=SRM_TORCH
   opticks_max_slot=M1
   opticks_event_mode=HitPhotonSeq

elif [ "$TEST" == "ref5" -o "$TEST" == "ref6" -o "$TEST" == "ref7" -o "$TEST" == "ref8" -o "$TEST" == "ref9" -o "$TEST" == "ref10" ]; then

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=M${TEST:3}
   opticks_running_mode=SRM_TORCH
   opticks_max_slot=M${TEST:3}

elif [ "$TEST" == "refX" ]; then

   opticks_num_event=1
   opticks_num_genstep=1
   opticks_num_photon=${X:-7500000}
   opticks_running_mode=SRM_TORCH
   opticks_max_slot=$opticks_num_photon

elif [ "$TEST" == "ref10_multilaunch" -o "$TEST" == "ref10_onelaunch" ]; then

   opticks_num_event=1
   opticks_num_genstep=10
   opticks_num_photon=M10
   opticks_running_mode=SRM_TORCH

   #opticks_max_photon=M10
   #opticks_max_curand=0    # zero loads all states : ready for whopper XORWOW running
   #opticks_max_curand=M10  # non-zero loads the specified number : this not relevant for PHILOX with default G1 1billion states

   case $TEST in
      *multilaunch) opticks_max_slot=M1 ;;     ## causes M10 to be done in 10 launches
        *onelaunch) opticks_max_slot=M10 ;;
   esac
   ## Normally leave max_slot as default zero indicating to pick max_slot according to VRAM.
   ## Are specifying here to compare multilaunch and onelaunch running of the same photons.

elif [ "$TEST" == "tiny_scan" ]; then

   opticks_num_event=10
   opticks_num_genstep=1x10
   opticks_num_photon=K1:10
   opticks_running_mode=SRM_TORCH
   #opticks_max_photon=M1

elif [ "$TEST" == "large_scan" ]; then

   opticks_num_event=20
   opticks_num_genstep=1x20
   opticks_num_photon=H1:10,M2,3,5,7,10,20,40,60,80,100
   opticks_running_mode=SRM_TORCH
   #opticks_max_photon=M100

elif [ "$TEST" == "medium_scan" ]; then

   opticks_num_event=12
   opticks_num_genstep=1x12
   opticks_num_photon=M1,1,10,20,30,40,50,60,70,80,90,100  # duplication of M1 is to workaround lack of metadata
   opticks_running_mode=SRM_TORCH
   #opticks_max_photon=M100

   # Remember multi-launch needs multiple gensteps in order to slice them up
   # such that each slice fits into VRAM.
   # So for big photon counts its vital to use multiple genstep.


elif [ "$TEST" == "larger_scan" ]; then

   opticks_num_event=22
   opticks_num_genstep=1x22
   opticks_num_photon=M1,1,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200  # duplication of M1 is to workaround lack of metadata
   opticks_running_mode=SRM_TORCH

   #opticks_max_photon=M200

elif [[ "$TEST" =~ ^v*large_evt$ ]]; then

   opticks_running_mode=SRM_TORCH
   opticks_num_event=1

   case $TEST in
        large_evt) opticks_num_genstep=10  ; opticks_num_photon=M200 ;;
       vlarge_evt) opticks_num_genstep=20  ; opticks_num_photon=M500 ;;
      vvlarge_evt) opticks_num_genstep=40  ; opticks_num_photon=G1   ;;
     vvvlarge_evt) opticks_num_genstep=120 ; opticks_num_photon=G3   ;;
    vvvvlarge_evt) opticks_num_genstep=256 ; opticks_num_photon=X32  ;;           # 4.29 billion
   vvvvvlarge_evt) opticks_num_genstep=512 ; opticks_num_photon=G5   ;;           #  5 billion
  vvvvvvlarge_evt) opticks_num_genstep=512 ; opticks_num_photon=8252787186   ;;   #  8.25 billion  https://www.worldometers.info/world-population/
   esac

   # KEEP_SUBFOLD DOUBLES SPACE AND TIME OF HIT SAVING
   #if [[ "$TEST" =~ ^v{4,}large_evt$ ]]; then
   #    export QSim__simulate_KEEP_SUBFOLD=1
   #fi


elif [ "$TEST" == "input_genstep" ]; then

   opticks_num_event=1000
   opticks_num_genstep=    # ignored
   opticks_num_photon=     # ignored ?
   opticks_running_mode=SRM_INPUT_GENSTEP

   #opticks_max_photon=M1

elif [ "$TEST" == "input_genstep_muon" ]; then

   opticks_num_event=1
   opticks_num_genstep=    # ignored
   opticks_num_photon=     # ignored ?
   opticks_running_mode=SRM_INPUT_GENSTEP

   #opticks_max_slot=H1     ##
   opticks_max_slot=M1     ##
   #opticks_max_slot=M5    ##
   #opticks_max_slot=M10   ## 3 launches
   #opticks_max_slot=M1    ## ~34 launches
   #opticks_max_slot=0     ## whole-in-one


elif [ "${TEST:0:12}" == "input_photon" ]; then

   opticks_num_event=1
   opticks_num_genstep=    # ignored
   opticks_num_photon=     # ignored ?
   opticks_running_mode=SRM_INPUT_PHOTON
   opticks_max_slot=M3

   if [ "${TEST:12}" == "_chimney" ]; then

      sevt__input_photon_dir=$TMP/SGenerate__test

      opticks_input_photon=SGenerate_ph_disc_K1.npy

      opticks_input_photon_frame=sChimneyLS:0:-2
      #opticks_input_photon_frame=sChimneyAcrylic

   elif [ "${TEST:12}" == "_nnvt" ]; then

      #sevt__input_photon_dir=/cvmfs/opticks.ihep.ac.cn/.opticks/InputPhotons
      sevt__input_photon_dir=$HOME/.opticks/InputPhotons

      opticks_input_photon=RainXZ_Z230_100k_f8.npy
      #opticks_input_photon=RainXZ_Z230_1000_f8.npy      ## ok
      #opticks_input_photon=RainXZ_Z230_10k_f8.npy       ## ok
      #opticks_input_photon=RainXZ_Z230_X700_10k_f8.npy  ## X700 to illuminate multiple PMTs

      opticks_input_photon_frame=NNVT:0:0
      #opticks_input_photon_frame=NNVT:0:50
      #opticks_input_photon_frame=NNVT:0:1000

   elif [ "${TEST:12}" == "_wp_pmt" ]; then

      sevt__input_photon_dir=$HOME/.opticks/InputPhotons
      opticks_input_photon=RainXZ_Z230_100k_f8.npy
      opticks_input_photon_frame=PMT_20inch_veto:0:1000

      #export PIDX=99999

   elif [ "${TEST:12}" == "_wp_pmt_side" ]; then

      sevt__input_photon_dir=$HOME/.opticks/InputPhotons
      opticks_input_photon=SideZX_X300_100k_f8.npy
      opticks_input_photon_frame=PMT_20inch_veto:0:1000

   elif [ "${TEST:12}" == "_wp_pmt_semi" ]; then

      sevt__input_photon_dir=$HOME/.opticks/InputPhotons
      opticks_input_photon=SemiCircleXZ_R-500_100k_f8.npy
      opticks_input_photon_frame=PMT_20inch_veto:0:1000

   elif [ "${TEST:12}" == "_poolcover" -o "${TEST:12}" == "_poolcover_refine" ]; then

      sevt__input_photon_dir=$HOME/.opticks/InputPhotons
      #opticks_input_photon=UpXZ1000_f8.npy
      opticks_input_photon=CircleXZ_R500_100k_f8.npy
      #opticks_input_photon=CircleXZ_R10_361_f8.npy
      #opticks_input_photon_frame=3345.569,20623.73,21500
      opticks_input_photon_frame=3345.569,20623.73,21000
      #export PIDX=99999

      export SEvt__transformInputPhoton_VERBOSE=1
      export CSGFoundry__getFrame_VERBOSE=1
      export CSGFoundry__getFrameE_VERBOSE=1

      if [ "${TEST/refine}" != "$TEST" ]; then
         export OPTICKS_PROPAGATE_REFINE=1
      else
         unset OPTICKS_PROPAGATE_REFINE
      fi


   elif [ "${TEST:12}" == "_s_pmt" ]; then

      sevt__input_photon_dir=$HOME/.opticks/InputPhotons
      opticks_input_photon=RainXZ_Z230_X25_100k_f8.npy
      opticks_input_photon_frame=PMT_3inch:0:0

   elif [ "${TEST:12}" == "_target" ]; then

       #opticks_input_photon=GridXY_X700_Z230_10k_f8.npy
       #opticks_input_photon=GridXY_X1000_Z1000_40k_f8.npy

       #opticks_input_photon_frame=-1
       #opticks_input_photon_frame=sWorld:0:0
       opticks_input_photon_frame=sTarget

   else
       echo $BASH_SOURCE : ERROR TEST [$TEST] SUFFIX AFTER input_photon [${TEST:12}] IS NOT HANDLED
       exit 1
   fi

else

   echo $BASH_SOURCE : ERROR TEST $TEST IS NOT HANDLED
   exit 1

fi

vars="$vars opticks_num_event opticks_num_genstep opticks_num_photon opticks_running_mode opticks_max_slot"


#opticks_running_mode=SRM_DEFAULT
#opticks_running_mode=SRM_INPUT_PHOTON
#opticks_running_mode=SRM_GUN


opticks_start_index=0
#opticks_max_bounce=31
opticks_max_bounce=63
opticks_integration_mode=1



#opticks_hit_mask=SD
opticks_hit_mask=EC
export OPTICKS_HIT_MASK=${OPTICKS_HIT_MASK:-$opticks_hit_mask}

export OPTICKS_NUM_EVENT=${OPTICKS_NUM_EVENT:-$opticks_num_event}
export OPTICKS_NUM_GENSTEP=${OPTICKS_NUM_GENSTEP:-$opticks_num_genstep}
export OPTICKS_NUM_PHOTON=${OPTICKS_NUM_PHOTON:-$opticks_num_photon}

export OPTICKS_RUNNING_MODE=${OPTICKS_RUNNING_MODE:-$opticks_running_mode}   # SRM_TORCH/SRM_INPUT_PHOTON/SRM_INPUT_GENSTEP
export OPTICKS_EVENT_MODE=${OPTICKS_EVENT_MODE:-$opticks_event_mode}         # what arrays are saved eg Hit,HitPhoton,HitPhotonSeq

export OPTICKS_MAX_PHOTON=${OPTICKS_MAX_PHOTON:-$opticks_max_photon}         # no needed much now with PHILOX and multi-launch

export OPTICKS_MAX_BOUNCE=${OPTICKS_MAX_BOUNCE:-$opticks_max_bounce}
export OPTICKS_START_INDEX=${OPTICKS_START_INDEX:-$opticks_start_index}
export OPTICKS_INTEGRATION_MODE=${OPTICKS_INTEGRATION_MODE:-$opticks_integration_mode}


vars="$vars version VERSION opticks_event_mode OPTICKS_EVENT_MODE OPTICKS_NUM_PHOTON OPTICKS_NUM_GENSTEP OPTICKS_MAX_PHOTON OPTICKS_NUM_EVENT OPTICKS_RUNNING_MODE"

export OPTICKS_MAX_CURAND=$opticks_max_curand  ## SEventConfig::MaxCurand only relevant to XORWOW
export OPTICKS_MAX_SLOT=$opticks_max_slot      ## SEventConfig::MaxSlot
vars="$vars OPTICKS_MAX_CURAND OPTICKS_MAX_SLOT"





if ! [ "$OPTICKS_EVENT_MODE" == "Minimal" -o "OPTICKS_EVENT_MODE" == "Hit" ]; then

    cat << EOW

WARNING : DEBUG RUNNING WITH OPTICKS_EVENT_MODE $OPTICKS_EVENT_MODE IS APPROPRIATE FOR SMALL STATISTICS ONLY

EOW
fi






if [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_GENSTEP" ]; then

    #igs=$BASE/jok-tds/ALL0/A000/genstep.npy
    #igs=$BASE/jok-tds/ALL0/A%0.3d/genstep.npy
    igs=$HOME/.opticks/crash_muon_igs.npy

    if [ "${igs/\%}" != "$igs" ]; then
        igs0=$(printf "$igs" 0)
    else
        igs0=$igs
    fi
    [ ! -f "$igs0" ] && echo $BASH_SOURCE : FATAL : NO SUCH PATH : igs0 $igs0 igs $igs && exit 1
    export OPTICKS_INPUT_GENSTEP=$igs
    export OPTICKS_START_INDEX=6   ## enabled reproducing photons from original eventID 6 from the igs


elif [ "$OPTICKS_RUNNING_MODE" == "SRM_INPUT_PHOTON" ]; then

    ## cf with ipcv : ~/j/InputPhotonsCheck/InputPhotonsCheck.sh
    # SEventConfig
    export SEvt__INPUT_PHOTON_DIR=${SEvt__INPUT_PHOTON_DIR:-$sevt__input_photon_dir}
    export OPTICKS_INPUT_PHOTON=${OPTICKS_INPUT_PHOTON:-$opticks_input_photon};
    export OPTICKS_INPUT_PHOTON_FRAME=${OPTICKS_INPUT_PHOTON_FRAME:-$opticks_input_photon_frame}

    ippath=${SEvt__INPUT_PHOTON_DIR}/${OPTICKS_INPUT_PHOTON}
    if [ ! -f "$ippath" ]; then
        echo $BASH_SOURCE - ERROR ippath [$ippath] DOES NOT EXIST
        exit 1
    fi



    vars="$vars SEvt__INPUT_PHOTON_DIR OPTICKS_INPUT_PHOTON OPTICKS_INPUT_PHOTON_FRAME"

elif [ "$OPTICKS_RUNNING_MODE" == "SRM_TORCH" ]; then

    #export SEvent_MakeGenstep_num_ph=100000  OVERRIDEN BY OPTICKS_NUM_PHOTON
    #export SEvent__MakeGenstep_num_gs=10     OVERRIDEN BY OPTICKS_NUM_GENSTEP

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
    #export SEvt__LIFECYCLE=1
    export SEvt__GATHER=1
    export SEvt__SAVE=1
}
[ -n "$LOG" ] && logging
[ -n "$LIFECYCLE" ] && export SEvt__LIFECYCLE=1
[ -n "$MEMCHECK" ] && export QU__MEMCHECK=1
[ -n "$MINIMAL"  ] && export SEvt__MINIMAL=1
[ -n "$MINTIME"  ] && export SEvt__MINTIME=1
[ -n "$INDEX"  ] && export SEvt__INDEX=1
[ -n "$RUNMETA"  ] && export SEvt__RUNMETA=1
[ -n "$CRASH" ] && export CSGOptiX__optixpath=$OPTICKS_PREFIX/ptx/objects-Debug/CSGOptiXOPTIX/CSGOptiX7.ptx

export QRng__init_VERBOSE=1
export SEvt__MINIMAL=1  ## just output dir
export SEvt__MINTIME=1  ## minimal timing info from QSim::simulate

#export SEvt__DIRECTORY=1  ## getDir dumping
#export SEvt__NPFOLD_VERBOSE=1
#export QSim__simulate_KEEP_SUBFOLD=1
#export SEvt__transformInputPhoton_VERBOSE=1
#export CSGFoundry__getFrameE_VERBOSE=1
#export CSGFoundry__getFrame_VERBOSE=1




if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%-30s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/env}" != "$arg" ]; then
    env | grep OPTICKS | perl -n -e 'm/(\S*)=(\S*)/ && printf("%50s : %s\n", $1, $2) ' -
fi

if [ "${arg/fold}" != "$arg" ]; then
    echo $AFOLD
    du -hs $AFOLD/*
fi

if [ "${arg/run}" != "$arg" -o "${arg/dbg}" != "$arg" ]; then

   knobs

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
       source dbg__.sh
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


if [ "${arg/deport}" != "$arg" ]; then
   source dbg__.sh
   dbg__ sreport
   [ $? -ne 0 ] && echo $BASH_SOURCE dreport error && exit 1
fi

if [ "${arg/report}" != "$arg" ]; then
   sreport
   [ $? -ne 0 ] && echo $BASH_SOURCE sreport error && exit 1
fi




if [ "${arg/grab}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $LOGDIR
fi

if [ "${arg/grep}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh ${LOGDIR}_sreport
fi

if [ "${arg/gevt}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $LOGDIR/$EVT
fi


if [ "${arg/du}" != "$arg" ]; then
    du -hs $AFOLD/*
fi


if [ "${arg/pdb}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi

if [ "${arg/pdz}" != "$arg" ]; then
    MODE=0 ${IPYTHON:-ipython} --pdb -i $script
fi

if [ "${arg/AB}" != "$arg" ]; then
    MODE=0 ${IPYTHON:-ipython} --pdb -i $script_AB
fi

if [ "${arg/ana}" != "$arg" ]; then
    MODE=0 ${PYTHON:-python} $script
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

