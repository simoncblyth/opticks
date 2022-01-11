#!/bin/bash -l 
msg="=== $BASH_SOURCE :"

#geom="dcyl_XZ"
#geom="bssc_XZ"
#geom="AdditionAcrylicConstruction_XZ"
#geom="BoxMinusTubs1_XZ"
#geom="SphereWithPhiSegment"
geom="AnnulusBoxUnion_YZ"

export GEOM=${GEOM:-$geom}
moi=0
dx=0
dy=0
dz=0
num_pho=100
isel=0   # setting isel to zero, prevents skipping bnd 0 
gridscale=0.1


dcyl(){ gridscale=0.025 ; }
bssc(){ gridscale=0.025 ; }
default()
{
    # everything else assume single PMT dimensions
    dz=-4
    isel=
    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
}


case $GEOM in 
   dcyl_*)                        cfbase=$TMP/CSGDemoTest/dcyl  && dcyl  ;;
   bssc_*)                        cfbase=$TMP/CSGDemoTest/bssc  && bssc  ;; 
   BoxMinusTubs_*)                cfbase=$TMP/GeoChain/BoxMinusTubs         ;;
   SphereWithPhiSegment_*)        cfbase=$TMP/GeoChain/SphereWithPhiSegment ;;
   AdditionAcrylicConstruction_*) cfbase=$TMP/GeoChain/AdditionAcrylicConstruction ;;
   BoxMinusTubs_*)                cfbase=$TMP/GeoChain/BoxMinusTubs ;;
   SphereWithPhiSegment_*)        cfbase=$TMP/GeoChain/SphereWithPhiSegment ;;
   AnnulusBoxUnion_*)             cfbase=$TMP/GeoChain/AnnulusBoxUnion ;;    
   *)                             cfbase=$TMP/GeoChain/$GEOM && default ;; 
esac

if [ "$GEOM" == "bssc_XZ" ]; then  
    note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects, maybe nice round demo numbers effect"
fi 

case $GEOM in  
   *XZ) cegs=16:0:9:$dx:$dy:$dz:$num_pho  ;;
   *YZ) cegs=0:16:9:$dx:$dy:$dz:$num_pho  ;;
   *XY) cegs=16:9:0:$dx:$dy:$dz:$num_pho  ;;
   *ZX) cegs=9:0:16:$dx:$dy:$dz:$num_pho  ;;
   *ZY) cegs=0:9:16:$dx:$dy:$dz:$num_pho  ;;
   *YX) cegs=9:16:0:$dx:$dy:$dz:$num_pho  ;;
esac

source ./cxs.sh 

