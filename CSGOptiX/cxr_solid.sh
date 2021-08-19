#!/bin/bash -l 

usage(){ cat << EOU
cxr_solid.sh
============

Single solid renders of standard solids selected by solid label eg r1@ 
the "@" is an alternate for "$" meaning to match the end of the string::

    ./cxr_solid.sh r0@
    ./cxr_solid.sh r1@
    ./cxr_solid.sh r2@
    ./cxr_solid.sh r3@
    ./cxr_solid.sh r4@

Renders selecting special debug ONE_PRIM_SOLID selected by solid labels starting with r1p r2p 
etc...  These typically select multiple solids that are presented with an IAS using 
Y shifts of twice the maximum extent of the selected solids.  Hence the below
show all the Prim in the corresponding standard solid arranged side by side::

    ./cxr_solid.sh r1p
    ./cxr_solid.sh r2p
    ./cxr_solid.sh r3p
    ./cxr_solid.sh r4p

Renders selecting the special debug ONE_NODE_SOLID selected by solid labels starting with R2P0N etc::

    ./cxr_solid.sh R2P0N
    ./cxr_solid.sh R8P0N11,R8P0N13,R8P0N14,R8P0N16,R8P0N17,R8P0N18,R8P0N19,R8P0N20,R8P0N21,R8P0N22,R8P0N15,R8P0N26

For multiple invokations of this script see::

    ./cxr_solids.sh

EOU
}

sla="${1:-r1p}"
eye="0,-5,0,1"
look="0,0,0,1"
tmin=0.1
zoom=1.0
quality=50

[ "${sla:(-1)}" == "@" ] && eye="0,-1,0,1"
[ "${sla:(-1)}" == "@" ] && tmin=1.0

if [ "$sla" == "r0@" ]; then 
   zoom=2.0
   quality=80
fi 



export SLA="$sla"
export CAM=1
export EYE=${EYE:-$eye} 
export LOOK=${LOOK:-$look}
export TMIN=${TMIN:-$tmin}
export ZOOM=${ZOOM:-$zoom}
export QUALITY=${QUALITY:-$quality}

#export GDB=lldb_ 

export NAMEPREFIX="cxr_solid_${sla}_"
export RELDIR=cxr_solid/cam_${CAM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_solid.sh $SLA      # EYE $EYE TMIN $TMIN  $stamp  $version " 

./cxr.sh  

exit 0
