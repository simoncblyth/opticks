#!/bin/bash
usage(){ cat << EOU
cxr_min.sh : Ray trace geometry rendering script
====================================================

Formerly described as "minimal script for shakedown"
but has become the first visualization script to use.
This uses one of two executables:

CSGOptiXRenderInteractiveTest
   interactive ray trace executable using OpenGL/CUDA interop

CSGOptiXRMTest
   single image ray trace executable with no OpenGL dependency is
   used when SNAP envvar is defined


Related for debug:

sysrap/tests/SGLM_set_frame_test.sh
   fast standalone SGLM::set_frame cycling using persisted sframe

Example commandlines using installed script::

    EYE=0.2,0.2,0.2 TMIN=0.1 cxr_min.sh
    EYE=0.3,0.3,0.3 TMIN=0.1 cxr_min.sh

    ELV=t:Water_solid,Rock_solid  cxr_min.sh

    MOI=PMT_20inch_veto:0:1000 cxr_min.sh


    EMM=2,3,4 EYE=3,3,0 cxr_min.sh


Use ELV to exclude virtual PMT wrapper volumes::

    ELV=t:HamamatsuR12860sMask_virtual,NNVTMCPPMTsMask_virtual,mask_PMT_20inch_vetosMask_virtual cxr_min.sh


EMM EnabledMergedMesh examples selecting compound solids::

    EMM=0, cxr_min.sh   ## only 0, "rem"

    EMM=t0, cxr_min.sh    ## exclude 0 "rem"
    EMM=t:0, cxr_min.sh   ## exclude 0 "rem"

    EMM=10, cxr_min.sh   ## only 10 "tri"

    EMM=10  cxr_min.sh   ## NB: WITHOUT COMMA IS VERY DIFFERENT TO ABOVE : BIT PATTERN FROM DECIMAL 10

    EMM=1,2,3,4 cxr_min.sh

    EMM=1,2,3,4,5,6,7,8,9 cxr_min.sh


NB presence of comma in EMM switches to bit position input, not binary value


EOU
}

SDIR=$(dirname $(realpath $BASH_SOURCE))

defarg=run_info
[ -n "$BP" ] && defarg=dbg_info
arg=${1:-$defarg}

bin=CSGOptiXRenderInteractiveTest
[ -n "$SNAP" ] && bin=CSGOptiXRMTest
which_bin=$(which $bin)

External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM
if [ -n "$GEOM" -a -n "${!External_CFBaseFromGEOM}" -a -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/CSGFoundry/prim.npy" ]; then
    echo $BASH_SOURCE - External GEOM setup detected
    vv="External_CFBaseFromGEOM ${External_CFBaseFromGEOM}"
    for v in $vv ; do printf "%40s : %s \n" "$v" "${!v}" ; done
else
    source ~/.opticks/GEOM/GEOM.sh  ## sets GEOM envvar, use GEOM bash function to setup/edit
fi

source $HOME/.opticks/GEOM/EVT.sh   ## optionally sets AFOLD BFOLD where event info is loaded from
source $HOME/.opticks/GEOM/MOI.sh   ## sets MOI envvar, controlling initial view, use MOI bash function to setup/edit



logging(){
   type $FUNCNAME
   #export CSGFoundry__Load_DUMP=1   # report the directory loaded
   #export CSGOptiX__prepareParamRender_DEBUG=1
   #export SGLM__updateProjection_DEBUG=1
   #export CSGFoundry=INFO
   #export CSGOptiX=INFO
   #export PIP=INFO
   #export SOpticksResource=INFO
   export SBT=INFO
}
[ -n "$LOG" ] && logging


#eye=1000,1000,1000
#eye=3.7878,3.7878,3.7878
#eye=-1,-1,0
#eye=-1,0,0
eye=1,0,0
#eye=3,0,0
#eye=3,0,-1.5
#eye=0,3,-1.5
#eye=0,-1,0
#eye=-1,-1,3
#eye=-1,-1,3

look=0,0,0

up=0,0,1
#up=0,0,-1    ## inverted is useful for PMT


wh=2560,1440
fullscreen=1

zoom=1
tmin=0.5
cam=perspective
#cam=orthographic   # needs work to set param to make view start closer to perspective

#escale=asis
escale=extent

traceyflip=0


## Formerly had some MOI dependent setting of
## view defaults, but as the view defaults apply
## to all frames have removed that kludge.

export WH=${WH:-$wh}
export FULLSCREEN=${FULLSCREEN:-$fullscreen}

export ESCALE=${ESCALE:-$escale}
export EYE=${EYE:-$eye}
export LOOK=${LOOK:-$look}
export UP=${UP:-$up}
export ZOOM=${ZOOM:-$zoom}
export TMIN=${TMIN:-$tmin}
export CAM=${CAM:-$cam}
export TRACEYFLIP=${TRACEYFLIP:-$traceyflip}


nameprefix=cxr_min_
nameprefix=${nameprefix}_eye_${EYE}_
nameprefix=${nameprefix}_zoom_${ZOOM}_
nameprefix=${nameprefix}_tmin_${TMIN}_

# moi appended within CSGOptiX::render_snap ?
export NAMEPREFIX=$nameprefix
## NAMEPREFIX used in CSGOptiX::getRenderStemDefault


topline="ESCALE=$ESCALE EYE=$EYE TMIN=$TMIN MOI=$MOI ZOOM=$ZOOM CAM=$CAM cxr_min.sh "
botline="$(date)"
export TOPLINE=${TOPLINE:-$topline}
export BOTLINE=${BOTLINE:-$botline}

## NB CUDA_VISIBLE_DEVICES is left up to the user to set - not opticks scripts

export CSGOptiXRenderInteractiveTest__FRAME_HOP=1
#export CSGOptiXRenderInteractiveTest__SGLM_DESC=1
export SGLFW__DEPTH=1   # dump _depth.jpg together with screenshots


TMP=${TMP:-/tmp/$USER/opticks}

export PBAS=${TMP}/    # note trailing slash
export BASE=$TMP/GEOM/$GEOM/$bin
export LOGDIR=$BASE
mkdir -p $LOGDIR
cd $LOGDIR

LOG=$bin.log

vars="bin which_bin GEOM MOI EMM ELV TMIN EYE LOOK UP ZOOM LOGDIR BASE PBAS NAMEPREFIX OPTICKS_HASH TOPLINE BOTLINE CUDA_VISIBLE_DEVICES"


Resolve_CFBaseFromGEOM()
{
   : LOOK FOR CFBase directory containing CSGFoundry geometry
   : HMM COULD PUT INTO GEOM.sh TO AVOID DUPLICATION ? BUT TOO MUCH HIDDEN ?
   : G4CXOpticks_setGeometry_Test GEOM TAKES PRECEDENCE OVER .opticks/GEOM
   : HMM : FOR SOME TESTS WANT TO LOAD GDML BUT FOR OTHERS CSGFoundry
   : to handle that added gdml resolution to eg g4cx/tests/GXTestRunner.sh

   local External_CFBaseFromGEOM=${GEOM}_CFBaseFromGEOM

   local A_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
   local B_CFBaseFromGEOM=$TMP/G4CXOpticks_setGeometry_Test/$GEOM
   local C_CFBaseFromGEOM=/cvmfs/opticks.ihep.ac.cn/.opticks/GEOM/$GEOM

   local TestPath=CSGFoundry/prim.npy
   local GDMLPathFromGEOM=$HOME/.opticks/GEOM/$GEOM/origin.gdml


    if [ -d "${!External_CFBaseFromGEOM}" -a -f "${!External_CFBaseFromGEOM}/$TestPath" ]; then
        echo $BASH_SOURCE : USING EXTERNALLY SETUP GEOMETRY ENVIRONMENT : EG FROM OJ DISTRIBUTION
    elif [ -d "$A_CFBaseFromGEOM" -a -f "$A_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$A_CFBaseFromGEOM
        echo $BASH_SOURCE : FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
    elif [ -d "$B_CFBaseFromGEOM" -a -f "$B_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$B_CFBaseFromGEOM
        echo $BASH_SOURCE : FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
    elif [ -d "$C_CFBaseFromGEOM" -a -f "$C_CFBaseFromGEOM/$TestPath" ]; then
        export ${GEOM}_CFBaseFromGEOM=$C_CFBaseFromGEOM
        echo $BASH_SOURCE : FOUND C_CFBaseFromGEOM $C_CFBaseFromGEOM containing $TestPath
    elif [ -f "$GDMLPathFromGEOM" ]; then
        export ${GEOM}_GDMLPathFromGEOM=$GDMLPathFromGEOM
        echo $BASH_SOURCE : FOUND GDMLPathFromGEOM $GDMLPathFromGEOM
    else
        echo $BASH_SOURCE : NOT-FOUND A_CFBaseFromGEOM $A_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND B_CFBaseFromGEOM $B_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND C_CFBaseFromGEOM $C_CFBaseFromGEOM containing $TestPath
        echo $BASH_SOURCE : NOT-FOUND GDMLPathFromGEOM $GDMLPathFromGEOM
    fi
}
Resolve_CFBaseFromGEOM





if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done
fi

if [ "${arg/run}" != "$arg" ]; then
   if [ -f "$LOG" ]; then
       echo $BASH_SOURCE : run : delete prior LOG $LOG
       rm "$LOG"
   fi
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1
fi

if [ "${arg/dbg}" != "$arg" ]; then
   if [ -f "$LOG" ]; then
       echo $BASH_SOURCE : dbg : delete prior LOG $LOG
       rm "$LOG"
   fi
   ## use installed script, not the source one so works from distribution
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 1
fi

if [ "$arg" == "grab" -o "$arg" == "open" -o "$arg" == "clean" -o "$arg" == "grab_open" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg
fi

if [ "${arg/list}" != "$arg" -o "${arg/pub}" != "$arg" ]; then
    source $OPTICKS_HOME/bin/BASE_grab.sh $arg
fi


