#!/usr/bin/env bash
cxr_min_anim_usage(){ cat << EOU
cxr_min_anim
==============

Script using cxr_min.sh to do record rendering.

* see GEOM and EVT for config of geometry and event


EOU
}


cxr_min_anim_setup(){
    type $FUNCNAME
    export SGLM_Option="A"  #  MO ABGMO
    #export CAM=orthographic
    export CAM=perspective
    #export ZOOMHOME=0.2
    export SGLM__renderloop_exit_DUMP=1

    #rec_shader_name=rec_flying_point_persist  # default
    rec_shader_name=rec_flying_point
    #rec_shader_name=rec_line_strip      # all those tightly packed lines mean need to reduce stats to be useful
    #rec_shader_name=rec_flying_vec
    export SGLFW_Evt__rec_shader_name=$rec_shader_name

    export SGLM__init_auxil=10,0,0,0  # Auxil uniform in rec geom.glsl

    export FULLSCREEN=0   # useful to see the incrementing sim time in window title
    export ANIM=1         # enable debug output regarding SRecord arrays and time cuts
    export T0=150     ## ns
    export T1=250     ## ns
    export TT=250     ## ns - alt reference time

    export TN=1000   ## larger slows down animation
}

cxr_min_anim_info()
{
    local vv="ANIM T0 T1 TT TN SGLM_Option CAM ZOOMHOME SGLM__renderloop_exit_DUMP SGLFW_Evt__rec_shader_name FULLSCREEN"
    for v in $vv ; do printf "%50s : %s\n" "$v" "${!v}" ; done
}

cxr_min_anim_main()
{
    local defarg=info_render
    local arg=${1:-$defarg}

    cxr_min_anim_setup

    if [[ "$arg" =~ help ]]; then
        cxr_min_anim_usage
    fi

    if [[ "$arg" =~ info ]]; then
        cxr_min_anim_info
    fi

    if [[ "$arg" =~ render ]]; then
        cxr_min.sh
        [ $? -ne 0 ] && echo $BASH_SOURCE - ERROR from render && exit 1
    fi
}

cxr_min_anim_main $*


