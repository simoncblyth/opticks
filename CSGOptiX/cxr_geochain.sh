#!/bin/bash -l
usage(){ cat << EOU
cxr_geochain.sh 
=================

Rendering CSGFoundry geometry created from
Geant4 C++ solid specification by GeoChain 
executable. Usage::

    cd ~/opticks/GeoChain   # gc 
    ./run.sh                # create geometry CF folder

    cd ~/opticks/CSGOptiX
    ./cxr_geochain.sh      # render jpg 
    EYE=0,0,1,1 ./cxr_geochain.sh  

The EYE, LOOK, UP envvars set the okc/View::home defaults 


DONE: avoid outdir having -1 elem in the path, as thats tedious to handle from shell 
DONE: add PUB=1 sensitivity to copy renders into standardized publication dirs based on the outdir 
      with the front of the path replaced 

Taking a look inside, hunting speckle::

    EYE=0,-0.7,0. UP=0,0,1 TMIN=0. CAM=0  ./cxr_geochain.sh 



Debugging blank renders::

     EYE=0.1,0.1,0.1 ./cxr_geochain.sh 
          shoot from inside so every pixel should intersect 
          add debug to trace whats happening 


EOU
}

msg="=== $BASH_SOURCE :"

#geom=default
#geom=AdditionAcrylicConstruction

#geom=pmt_solid
#geom=1_3

#geom=UnionOfHemiEllipsoids        # looks fine, like full ellipsoid
#geom=UnionOfHemiEllipsoids-50    # lower hemi-ellipsoid is smaller than upper : looks like the translation transform stomps on the scale transform

#geom=body_solid
#geom=inner_solid
#geom=inner1_solid
#geom=inner2_solid

#geom=body_phys
#geom=inner1_phys
#geom=inner2_phys

#geom=SphereWithPhiSegment
#geom=Orb
#geom=PolyconeWithMultipleRmin

#geom=hmsk_solidMask
#geom=hmsk_solidMaskTail

#geom=nmsk_solidMask
#geom=nmsk_solidMaskTail

#geom=XJfixtureConstruction
#geom=AltXJfixtureConstruction
#geom=XJanchorConstruction
#geom=AnnulusBoxUnion
#geom=AnnulusTwoBoxUnion
#geom=AnnulusFourBoxUnion
#geom=AnnulusOtherTwoBoxUnion

#geom=BoxFourBoxUnion
#geom=BoxFourBoxContiguous

geom=SphereWithPhiCutDEV

catgeom=$(cat ~/.opticks/GEOM.txt 2>/dev/null) && [ -n "$catgeom" ] && echo $msg catgeom $catgeom override of default geom $geom && geom=$catgeom
export GEOM=${GEOM:-$geom}

# cfname picks the CSGFoundry geometry to load
if [ "$(uname)" == "Linux" ]; then
    cfname=GeoChain/$GEOM            
else
    cfname=GeoChain_Darwin/$GEOM            
fi


moi=ALL
zoom=1

if [ "$GEOM" == "default" ]; then  
   eye=-1,0,1,1 
   tmin=0.5
   cam=0         
elif [ "$GEOM" == "XJfixtureConstruction" ]; then  
   eye=1,1,1
   tmin=0.1 
   cam=1
   zoom=2
else
   eye=-2,0,0,1        
   tmin=1
   cam=1     # 0:perspective  1:ortho        
fi 

emm=t0    # default to no solid skips with GeoChain, which is typically single solid geometry 

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)
export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
export EMM=${EMM:-$emm}
export TMIN=${TMIN:-$tmin}
export EYE=${EYE:-$eye}
export CAM=${CAM:-$cam}    # 0:perspective 1:ortho
export ZOOM=${ZOOM:-$zoom}

export NAMEPREFIX=cxr_geochain_${GEOM}_   # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_geochain/cam_${CAM}
export TOPLINE="./cxr_geochain.sh $MOI      # EYE $EYE  $stamp  $version $GEOM   " 

vars="GEOM MOI CFNAME EMM TMIN EYE CAM NAMEPREFIX RELDIR TOPLINE"
echo $msg 
for var in $vars ; do printf "%-20s : %s \n" $var "${!var}" ; done  

echo $msg invoke ./cxr.sh 
./cxr.sh 

exit 0

