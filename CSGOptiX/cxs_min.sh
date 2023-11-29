#!/bin/bash -l 
usage(){ cat << EOU
cxs_min.sh : minimal executable and script for shakedown
============================================================

Usage::


     ~/opticks/CSGOptiX/cxs_min.sh
     ~/opticks/CSGOptiX/cxs_min.sh info
     ~/opticks/CSGOptiX/cxs_min.sh run
     ~/opticks/CSGOptiX/cxs_min.sh report


    BP=SEvt::SEvt ~/opticks/CSGOptiX/cxs_min.sh
    BP=SEvent::MakeTorchGenstep ~/opticks/CSGOptiX/cxs_min.sh

    MODE=2 SEL=1 ~/opticks/CSGOptiX/cxs_min.sh ana 

    EVT=p005 ~/opticks/CSGOptiX/cxs_min.sh ana 
    EVT=p010 ~/opticks/CSGOptiX/cxs_min.sh ana


EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

case $(uname) in
   #Linux) defarg=dbg_info ;;
   Linux) defarg=run_report_info ;;
   Darwin) defarg=ana ;;
esac

if [ -n "$BP" ]; then
   defarg="dbg"
fi 

arg=${1:-$defarg}

#export OPTICKS_HASH=$(git -C $OPTICKS_HOME rev-parse --short HEAD)
# assumes source available

bin=CSGOptiXSMTest
script=$SDIR/cxs_min.py

source ~/.opticks/GEOM/GEOM.sh   # sets GEOM envvar 

export EVT=${EVT:-p001}
export BASE=${TMP:-/tmp/$USER/opticks}/GEOM/$GEOM
export BINBASE=$BASE/$bin

#export VERSION=0   # StandardFullDebug
#export VERSION=1   # Minimal  

export LOGDIR=$BINBASE/ALL${VERSION:-0}
export AFOLD=$BINBASE/ALL${VERSION:-0}/$EVT

#export BFOLD=$BASE/G4CXTest/ALL0/$EVT  ## comparison with "A" from another executable
export BFOLD=$BASE/jok-tds/ALL0/p001    ## comparison with "A" from another executable


mkdir -p $LOGDIR 
cd $LOGDIR 
LOGFILE=$bin.log


#srm=SRM_DEFAULT
srm=SRM_TORCH
#srm=SRM_INPUT_PHOTON
#srm=SRM_INPUT_GENSTEP
#srm=SRM_GUN
export OPTICKS_RUNNING_MODE=$srm

echo $BASH_SOURCE OPTICKS_RUNNING_MODE $OPTICKS_RUNNING_MODE

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

    onp=K1:10 
    #onp=H1:10  
    export OPTICKS_NUM_PHOTON=${ONP:-$onp}
    export SEvent_MakeGenstep_num_ph=100000
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



#oem=StandardFullDebug
oem=Minimal
export OPTICKS_EVENT_MODE=${OEM:-$oem}
export OPTICKS_MAX_BOUNCE=31
export OPTICKS_MAX_PHOTON=1000000
export OPTICKS_INTEGRATION_MODE=1
export OPTICKS_NUM_EVENT=10

cvd=1   # default 1:TITAN RTX
export CUDA_VISIBLE_DEVICES=${CVD:-$cvd}


logging(){ 
    export CSGFoundry=INFO
    export CSGOptiX=INFO
    export QEvent=INFO 
    export QSim=INFO
    #export SEvt__DEBUG_CLEAR=1   # see ~/opticks/notes/issues/SEvt__clear_double_call.rst
    #export SEvt__LIFECYCLE=1
}
[ -n "$LOG" ] && logging


vars="GEOM LOGDIR BINBASE OPTICKS_HASH CVD CUDA_VISIBLE_DEVICES SDIR FOLD LOG NEVT"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done 
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
       $bin
   elif [ "${arg/dbg}" != "$arg" ]; then
       dbg__ $bin 
   fi 
   [ $? -ne 0 ] && echo $BASH_SOURCE run/dbg error && exit 1 
fi 

if [ "${arg/report}" != "$arg" ]; then
   sreport
   [ $? -ne 0 ] && echo $BASH_SOURCE sreport error && exit 1 

   sprof_fold_report
   [ $? -ne 0 ] && echo $BASH_SOURCE sprof_fold_report error && exit 1 
fi 

if [ "${arg/grab}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/rsync.sh $BINBASE
fi 

if [ "${arg/ana}" != "$arg" ]; then
    ${IPYTHON:-ipython} --pdb -i $script
fi 


