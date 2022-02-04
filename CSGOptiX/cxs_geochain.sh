#!/bin/bash -l 
usage(){ cat << EOU

NOMASK=1 ./cxs_geochain.sh 
   use NOMASK to debug empty frames, eg when the genstep grid is too small for the geometry 


MASK=t GEOM=AltXJfixtureConstruction_XYZ ./cxs_geochain.sh 
   3D pyvista view of intersects  

GEOM=AltXJfixtureConstruction_XY ./cxs_geochain.sh 


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
#geom="AnnulusFourBoxUnion_YZ"

#geom="AnnulusOtherTwoBoxUnion_XY"
#geom="AnnulusOtherTwoBoxUnion_XZ"

#geom="AltXJfixtureConstruction_YZ"
#geom="AltXJfixtureConstruction_XZ"
#geom="AltXJfixtureConstruction_XY"
geom="AltXJfixtureConstruction_XYZ"

geom="BoxFourBoxContiguous_YX"


export GEOM=${GEOM:-$geom}
moi=0   # intended to catch the first meshname (which for geochain is usually the only meshname)
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
default(){ echo -n  ; }

pmt_default()
{
    # everything else assume single PMT dimensions
    dz=-4
    isel=
    unset CXS_OVERRIDE_CE
    export CXS_OVERRIDE_CE=0:0:-130:320   ## fix at the full uncut ce 
}

gcn=${GEOM%%_*}  ## name up to the first underscore, assuming use of axis suffix  _XZ _YZ _XY _ZX _ZY _YX 


case $GEOM in 
   dcyl_*)    cfbase=$TMP/CSGDemoTest/$gcn  && dcyl     ;;
   bssc_*)    cfbase=$TMP/CSGDemoTest/$gcn  && bssc     ;; 
   Annulus*)  cfbase=$TMP/GeoChain/$gcn     && Annulus  ;;    
   *)         cfbase=$TMP/GeoChain/$gcn     && default  ;;    
esac


case $GEOM in 
   bssc_XZ) note="HMM : box minus sub-sub cylinder NOT showing the spurious intersects, maybe nice round demo numbers effect" ;; 
   AnnulusBoxUnion_YZ) note="no spurious intersects seen" ;; 
   AnnulusBoxUnion_XY) note="no spurious intersects seen" ;; 
   AnnulusTwoBoxUnion_XY) note="no spurious intersects seen" ;; 
   AnnulusTwoBoxUnion_YZ) note="no spurious" ;; 
   AnnulusFourBoxUnion_XY) note="spurious intersects appear with four boxes, not with two" ;; 
   AnnulusFourBoxUnion_YZ) note="curious the spurious intersects visible in XY cross-section are not apparent in YZ cross-section" ;; 
   AnnulusOtherTwoBoxUnion_XY) note="no spurious intersects seen" ;; 
   AnnulusOtherTwoBoxUnion_XZ) note="no spurious intersects seen" ;; 
   AltXJfixtureConstruction_YZ) note="spurious intersects in YZ plane avoided with the Alt CSG modelling" ;; 
   AltXJfixtureConstruction_XZ) note="thin xbox cross piece apparent" ;; 
   AltXJfixtureConstruction_XY) note="some spurious remain between the curve of the outer tubs and the protruding boxes" ;; 
esac

case $GEOM in  
   *_XZ) cegs=16:0:9:$dx:$dy:$dz:$num_pho  ;;
   *_YZ) cegs=0:16:9:$dx:$dy:$dz:$num_pho  ;;
   *_XY) cegs=16:9:0:$dx:$dy:$dz:$num_pho  ;;
   *_ZX) cegs=9:0:16:$dx:$dy:$dz:$num_pho  ;;
   *_ZY) cegs=0:9:16:$dx:$dy:$dz:$num_pho  ;;
   *_YX) cegs=9:16:0:$dx:$dy:$dz:$num_pho  ;;
   *_XYZ) cegs=9:16:9:$dx:$dy:$dz:$num_pho ;;  
esac
# first axis named is the longer one that is presented on the horizontal in landscape aspect   

echo $msg GEOM $GEOM gcn $gcn cegs $cegs cfbase $cfbase


source ./cxs.sh 


