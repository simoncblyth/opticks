#!/bin/bash -l 

usage(){ cat << EOU
cxr_solid.sh
============

::

    SLA=r0@ EYE=0,-0.5,0.75 TMIN=0.5 ./cxr_solid.sh 


Single solid renders of standard solids selected by solid label eg r1@ 
the "@" is an alternate for "$" meaning to match the end of the string::

    SLA=r0@ ./cxr_solid.sh 
    SLA=r1@ ./cxr_solid.sh 
    SLA=r2@ ./cxr_solid.sh 
    SLA=r3@ ./cxr_solid.sh 
    SLA=r4@ ./cxr_solid.sh 


For multiple invokations of this script see::

    ./cxr_solids.sh


Below need a non-standard CSGFoundry geometry and have not been tested recently
---------------------------------------------------------------------------------

Renders selecting special debug ONE_PRIM_SOLID selected by solid labels starting with r1p r2p 
etc...  These typically select multiple solids that are presented with an IAS using 
Y shifts of twice the maximum extent of the selected solids.  Hence the below
show all the Prim in the corresponding standard solid arranged side by side::

    SLA=r1p ./cxr_solid.sh
    SLA=r2p ./cxr_solid.sh
    SLA=r3p ./cxr_solid.sh

Renders selecting the special debug ONE_NODE_SOLID selected by solid labels starting with R2P0N etc::

    SLA=R2P0N ./cxr_solid.sh 
    SLA=R8P0N11,R8P0N13,R8P0N14,R8P0N16,R8P0N17,R8P0N18,R8P0N19,R8P0N20,R8P0N21,R8P0N22,R8P0N15,R8P0N26  ./cxr_solid.sh 



EOU
}

sla="r0@"
export SLA=${SLA:-$sla}

eye="0,-5,0,1"
look="0,0,0,1"
tmin=0.1
zoom=1.0
quality=50
cam=1         # 0:perspective 1:ortho 

[ "${SLA:(-1)}" == "@" ] && eye="0,-1,0,1"
[ "${SLA:(-1)}" == "@" ] && tmin=1.0

if [ "$SLA" == "r0@" ]; then 
   zoom=2.0
   quality=80
fi 

export EYE=${EYE:-$eye} 
export LOOK=${LOOK:-$look}
export ZOOM=${ZOOM:-$zoom}
export TMIN=${TMIN:-$tmin}

nameprefix=cxr_solid_${SLA}_

if [ "$EYE" != "$eye" ]; then 
   nameprefix=${nameprefix}_eye_${EYE}_
fi 
if [ "$LOOK" != "$look" ]; then 
   nameprefix=${nameprefix}_look_${LOOK}_
fi 
if [ "$ZOOM" != "$zoom" ]; then 
   nameprefix=${nameprefix}_zoom_${ZOOM}_
fi 
if [ "$TMIN" != "$tmin" ]; then 
   nameprefix=${nameprefix}_tmin_${TMIN}_
fi 

export CAM=${CAM:-$cam}
export QUALITY=${QUALITY:-$quality}

#export GDB=lldb_ 

export NAMEPREFIX=$nameprefix
export OPTICKS_RELDIR=cam_${CAM}
export OPTICKS_GEOM=cxr_solid_$SLA

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)

export TOPLINE="./cxr_solid.sh $SLA      # EYE $EYE TMIN $TMIN  $stamp  $version " 

./cxr.sh  

exit 0
