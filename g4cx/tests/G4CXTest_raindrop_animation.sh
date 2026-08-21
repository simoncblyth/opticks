#!/bin/bash
usage(){ cat << EOU
G4CXTest_raindrop_animation.sh
================================

* DONE: revived record animation

  * checked ~/o/examples/UseGeometryShader - nothing wrong with machinery - need to debug time ranges etc
  * issue was overlarge (20G) record array without record slicing configured

* TODO: debug why compositing of ray trace geom and event record not fully working,
  see many more photon records when switch off geometry render with alt+O


alt-A
   enable photon record aimimation rendering of AFOLD/record.npy
alt-B
   enable photon record aimimation rendering of BFOLD/record.npy


alt-T
   reset time back to T0

ctrl-T
   toggle animation time progression - ie stop/start time


O
   toggle orthographic/perspective projection

W/S
   forward backwards viewpoint control, does nothing in orthographic mode

Z
   toggle zoom control - then drag mouse up/down to adjust,
   zoon often needed in orthographic mode to make all geom visible

alt-O
   toggle rendering of geometry - very useful to make record points more visible
   (seems some bug with compositing)



EOU
}


source /data1/blyth/local/opticks_Debug/envset.sh

# NB the below standard scripts which are sourced by cxr_min.sh control crucial config settings
# so it does not make sense setting most things here
#
#
# ~/.opticks/GEOM/GEOM.sh  # geometry to load
# ~/.opticks/GEOM/MOI.sh   # viewpoint within geometry
# ~/.opticks/GEOM/EVT.sh   # record arrays to load - setting AFOLD, BFOLD and slicing



raindrop_anim(){
    : TODO - MOVE THIS FUNC TO COMMON LOCATION IN REPO THAT GETS INSTALLED - TO AVOID DUPLICATION WITH sysrap ssst.sh
    type $FUNCNAME
    export SGLM_Option="A"  #  MO ABGMO
    export CAM=orthographic
    #export CAM=perspective
    export ZOOMHOME=0.2
    export SGLM__renderloop_exit_DUMP=1


    #rec_shader_name=rec_flying_point_persist  # default
    #rec_shader_name=rec_flying_point
    #rec_shader_name=rec_line_strip      # all those tightly packed lines mean need to reduce stats to be useful
    rec_shader_name=rec_flying_vec
    export SGLFW_Evt__rec_shader_name=$rec_shader_name

    export SGLM__init_auxil=0.01,0,0,0  # Auxil uniform in rec geom.glsl

    export FULLSCREEN=0   # useful to see the incrementing sim time in window title
    export ANIM=1         # enable debug output regarding SRecord arrays and time cuts
    export T0=0      ## ns
    export T1=1.5    ## ns
    export TT=0.5    ## ns - alt reference time

    export TN=5000   ## larger slows down animation
}
raindrop_anim


defarg=info_render
arg=${1:-$defarg}

vv="ANIM T0 T1 TT TN"

if [[ "$arg" =~ help ]]; then
    usage
fi

if [[ "$arg" =~ info ]]; then
    for v in $vv ; do printf "%50s : %s\n" "$v" "${!v}" ; done
fi

if [[ "$arg" =~ render ]]; then
    cxr_min.sh
    [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from render && exit 1
fi

exit 0

