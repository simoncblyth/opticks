#!/bin/bash -l 

usage(){ cat << EOU
cxs_debug.sh
=============

See notes/issues/cxsim-shakedown.rst 

EOU
}


moi=37684
ce_offset=0
ce_scale=1   
cegs=16:0:9:500   
gridscale=0.10

source ./cxs.sh $*

