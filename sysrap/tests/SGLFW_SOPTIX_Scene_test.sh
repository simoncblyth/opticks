#!/bin/bash
usage(){ cat << EOU
SGLFW_SOPTIX_Scene_test.sh : triangulated raytrace and rasterized visualization
=================================================================================

Assuming the scene folder exists already::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test.sh

As this uses GL interop it may be necessary to select the display GPU, eg with::

    export CUDA_VISIBLE_DEVICES=1


Bash args
------------

info
   dump vars

open
   write context file, use this to control where screenshots are copied to

run
   run executable

dbg
   run under gdb

close
   delete context file


ENVVAR notes
--------------

WH FULLSCREEN EYE LOOK UP TMIN ESCALE CAM VIZMASK
    view control envvars used by SGLM.h, details for some of them below

VIZMASK
    control CSGSolid to display, see SGLM::is_vizmask_set and SOPTIX_Properties::visibilityMask
    examples::

         VIZMASK=0      # only global:0
         VIZMASK=0,1
         VIZMASK=t0     # mask global:0 (t0 means NOT 0)
         VIZMASK=t0,1   # mask global:0 and 1


SGLFW__DEPTH
    enables download and saving of jpg depth maps together with ordinary screenshots
    see SGLFW::init_img_frame

SOPTIX__HANDLE
    selects what to include in the OptiX geometry, not the OpenGL one,
    so must switch to the CUDA/OptiX render with C key to see effect of changes.
    As solids usually much smaller than full geometry will usually need to
    adjust the target frame, eg with::

        MOI=EXTENT:1000 SGLFW_SOPTIX_Scene_test.sh

SOPTIX_Options__LEVEL
    SOPTIX logging level


Non-CMake local build for development
--------------------------------------

This script used the CMake built binary.
For a separate non-CMake built binary see::

    ~/o/sysrap/tests/SGLFW_SOPTIX_Scene_test_local.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=SGLFW_SOPTIX_Scene_test
bin=$name
export SCRIPT=$name

source $HOME/.opticks/GEOM/GEOM.sh
[ -z "$GEOM" ] && echo $BASH_SOURCE FATAL GEOM $GEOM IS REQUIRTED && exit 1

_CFB=${GEOM}_CFBaseFromGEOM
if [ ! -d "${!_CFB}/CSGFoundry/SSim/scene" ]; then
   echo $BASH_SOURCE : FATAL GEOM $GEOM ${_CFB} ${!_CFB}
   exit 1
fi

source $HOME/.opticks/GEOM/EVT.sh 2>/dev/null  ## optionally sets AFOLD BFOLD where event info is loaded from
source $HOME/.opticks/GEOM/MOI.sh 2>/dev/null  ## optionally sets MOI envvar controlling initial viewpoint
source $HOME/.opticks/GEOM/ELV.sh 2>/dev/null  ## optionally set ELV envvar controlling included/excluded LV by name
source $HOME/.opticks/GEOM/SDR.sh 2>/dev/null  ## optionally configure OpenGL shader
source $HOME/.opticks/GEOM/CUR.sh 2>/dev/null  ## optionally define CUR_ bash function

logging()
{
   type $FUNCNAME

   #export SScene__level=1
   #export SGLFW_Scene__DUMP=1
   #export SGLM__set_frame_DUMP=1
   #export SGLFW_SOPTIX_Scene_test_DUMP=1
   #export SOPTIX_SBT__initHitgroup_DUMP=1
   #export SOPTIX_Options__LEVEL=0
   #export SOPTIX_Scene__DUMP=1
}
[ -n "$LOG" ] && logging

anim()
{
   type $FUNCNAME
   export SGLM__init_time_DUMP=1
}
[ -n "$ANIM" ] && anim

[ -n "$OPENGL" ] && export SGLFW_check__level=1



if [ -f "$HOME/.opticks/GEOM/VUE.sh" ]; then
    source $HOME/.opticks/GEOM/VUE.sh
else

    #wh=1024,768
    wh=2560,1440

    #fullscreen=0
    fullscreen=1

    #eye=0.1,0,-10
    #eye=-1,-1,0
    #eye=-10,-10,0
    #eye=-10,0,0
    #eye=0,-10,0
    #eye=-1,-1,0
    eye=0,1,0

    look=0,0,0
    up=0,0,1

    tmin=0.1

    #escale=asis
    escale=extent

    cam=perspective
    #cam=orthographic

    vizmask=t    # 0xff default, no masking
    #vizmask=t0  # 0xfe mask global, only instanced geometry in both OpenGL and OptiX renders

    export WH=${WH:-$wh}
    export FULLSCREEN=${FULLSCREEN:-$fullscreen}
    export EYE=${EYE:-$eye}
    export LOOK=${LOOK:-$look}
    export UP=${UP:-$up}
    export TMIN=${TMIN:-$tmin}
    export ESCALE=${ESCALE:-$escale}
    export CAM=${CAM:-$cam}

    ## VIZMASK may be ssst.sh ONLY ?
    export VIZMASK=${VIZMASK:-$vizmask}

    #export CSGOptiXRenderInteractiveTest__SGLM_DESC=1
    #export SGLFW__DEPTH=1   # dump _depth.jpg together with screenshots

    soptix__handle=-1  # default, full geometry
    #soptix__handle=0  #  only non-instanced global geometry
    #soptix__handle=1  #  single CSGSolid
    #soptix__handle=2  #
    export SOPTIX__HANDLE=${SOPTIX__HANDLE:-$soptix__handle}

    #export SOPTIX_Options__LEVEL=1

fi



_CUR=GEOM/$GEOM/$SCRIPT/$EVT_CHECK


allarg="info_open_dbg_run_close_touch"
defarg="info_open_run"
[ -n "$BP" ] && defarg="info_dbg"

arg=${1:-$defarg}
arg2=$2

vars="BASH_SOURCE allarg defarg arg name bin GEOM _CUR"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done
    printf "\n"
fi

if [ "${arg/dbg}" != "$arg" ]; then
    source dbg__.sh
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 2
fi

if [ "${arg/open}" != "$arg" ]; then
    # open to define current context string which controls where screenshots are copied to
    CUR_open ${_CUR}
fi

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/close}" != "$arg" ]; then
    # close to invalidate the context
    CUR_close
fi

if [ "$arg" == "touch"  ]; then
    if [ -n "$arg2" ]; then
        CUR_touch "$arg2"
    else
        echo $BASH_SOURCE:touch needs arg2 datetime accepted by CUR_touch eg "ssst.sh touch 11:00"
    fi
fi

exit 0

