#!/bin/bash -l 
usage(){ cat << EOU

NOMASK=1 ./cxs_geochain.sh 
    use NOMASK to debug empty frames, eg when the genstep grid is too small for the geometry 

EOU
}



msg="=== $BASH_SOURCE :"

#geom="dcyl_XZ"
#geom="bssc_XZ"
#geom="AdditionAcrylicConstruction_XZ"
#geom="BoxMinusTubs1_XZ"
#geom="SphereWithPhiSegment"
#geom="AnnulusBoxUnion_XY"
#geom="AnnulusBoxUnion_YZ"

#geom="AnnulusTwoBoxUnion_XY"
#geom="AnnulusTwoBoxUnion_YZ"
#geom="AnnulusFourBoxUnion_XY"
geom="AnnulusFourBoxUnion_YZ"


export GEOM=${GEOM:-$geom}
moi=0
dx=0
dy=0
dz=0
num_pho=100
isel=0   # setting isel to zero, prevents skipping bnd 0 
gridscale=0.1
ce_offset=0
ce_scale=1
gsplot=1


dcyl(){    gridscale=0.025 ; }
bssc(){    gridscale=0.025 ; }
Annulus(){ gridscale=0.15 ;  }  # enlarge genstep grid to fit the protruding unioned boxes

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
   AnnulusBoxUnion_*)             cfbase=$TMP/GeoChain/AnnulusBoxUnion         && Annulus ;;    
   AnnulusTwoBoxUnion_*)          cfbase=$TMP/GeoChain/AnnulusTwoBoxUnion      && Annulus ;;    
   AnnulusOtherTwoBoxUnion_*)     cfbase=$TMP/GeoChain/AnnulusOtherTwoBoxUnion && Annulus ;;    
   AnnulusFourBoxUnion_*)         cfbase=$TMP/GeoChain/AnnulusFourBoxUnion     && Annulus ;;    
   *)                             cfbase=$TMP/GeoChain/$GEOM && default ;; 
esac


case $GEOM in 
   bssc_XZ) note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects, maybe nice round demo numbers effect" ;; 
   AnnulusBoxUnion_YZ) note="no spurious intersects seen" ;; 
   AnnulusBoxUnion_XY) note="no spurious intersects seen" ;; 
   AnnulusTwoBoxUnion_XY) note="no spurious intersects seen" ;; 
   AnnulusTwoBoxUnion_YZ) note="no spurious" ;; 
   AnnulusFourBoxUnion_XY) note="spurious intersects appear with four boxes, not with two" ;; 
   AnnulusFourBoxUnion_YZ) note="curious the spurious intersects visible in XY cross-section are not apparent in YZ cross-section" ;; 
esac

case $GEOM in  
   *XZ) cegs=16:0:9:$dx:$dy:$dz:$num_pho  ;;
   *YZ) cegs=0:16:9:$dx:$dy:$dz:$num_pho  ;;
   *XY) cegs=16:9:0:$dx:$dy:$dz:$num_pho  ;;
   *ZX) cegs=9:0:16:$dx:$dy:$dz:$num_pho  ;;
   *ZY) cegs=0:9:16:$dx:$dy:$dz:$num_pho  ;;
   *YX) cegs=9:16:0:$dx:$dy:$dz:$num_pho  ;;
esac
# first axis named is the longer one that is presented on the horizontal in landscape aspect   

source ./cxs.sh 


