#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

old_src_base=$TMP/CSGOptiX/CSGOptiXSimulateTest


default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}
new_src_base=$HOME/$opticks_keydir_grabbed/CSG_GGeo/CSGOptiXSimulateTest

export SRC_BASE=$new_src_base

#echo $msg SRC_BASE $SRC_BASE


pub.py $* 

