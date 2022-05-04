#!/bin/bash -l
usage(){ cat << EOU
cxr_geochain.sh : 3D renders using CSGOptiXRenderTest
==========================================================

Rendering CSGFoundry geometry created from
Geant4 C++ solid specification by GeoChain 
executable. 

Usage:

1. create a CSGFoundry geometry folder by conversion from a Geant4 geometry using GeoChain::

    cd ~/opticks/GeoChain   # gc 
    ./translate.sh                # create geometry CF folder

2. alternatively some geometry is only implemnented at CSG level, create CSGFoundry with::

    cd ~/opticks/CSG   # c 
    ./CSGMakerTest.sh 

3. run the render, this works with OptiX 7 (needs manual b7 build) and also by virtue of CSGOptiX/Six.cc with OptiX 5 or 6:: 
 
    cd ~/opticks/CSGOptiX    # cx

    ./cxr_geochain.sh                 # render jpg 
    EYE=0,0,1,1 ./cxr_geochain.sh     # change viewpoint 

The EYE, LOOK, UP envvars set the okc/View::home defaults 

DONE: avoid outdir having -1 elem in the path, as thats tedious to handle from shell 
DONE: add PUB=1 sensitivity to copy renders into standardized publication dirs based on the outdir 
      with the front of the path replaced 

Taking a look inside, hunting speckle::

    EYE=0,-0.7,0. UP=0,0,1 TMIN=0. CAM=0  ./cxr_geochain.sh 


Debugging unexpected renders
------------------------------

1. force recompile the kernels by touching .cu 

   * have observed that changes to CSG headers do not automatically 
     trigger rebuilds of the .cu kernels : so touch the .cu to 
     make sure get the latest CSG headers


Debugging blank renders
------------------------

Arrange a viewpoint from the inside of a shape such that every pixel should intersect. 
Thence can trace whats happening with any pixel. 

::

     EYE=0.1,0.1,0.1 ./cxr_geochain.sh 
          shoot from inside so every pixel should intersect 
          add debug to trace whats happening 


EYE=1,1,1 GEOM=DifferenceBoxSphere TMIN=0.5 ./cxr_geochain.sh 


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

#geom=SphereWithPhiCutDEV
#geom=GeneralSphereDEV
#geom=OverlapBoxSphere
#geom=ContiguousBoxSphere
#geom=DiscontiguousBoxSphere

#geom=ContiguousThreeSphere
#geom=OverlapThreeSphere
#geom=parade
# export GEOM=${GEOM:-$geom}

source $PWD/../bin/GEOM.sh trim   ## sets GEOM envvar 

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
   icam=0         
elif [ "$GEOM" == "XJfixtureConstruction" ]; then  
   eye=1,1,1
   tmin=0.1 
   icam=1
   zoom=2
else
   eye=-2,0,0,1        
   tmin=1
   icam=1     # 0:perspective  1:ortho        
fi 

emm=t0    # default to no solid skips with GeoChain, which is typically single solid geometry 

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion 2>/dev/null)
export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
export EMM=${EMM:-$emm}
export TMIN=${TMIN:-$tmin}
export EYE=${EYE:-$eye}
export ICAM=${ICAM:-$icam}    # 0:perspective 1:ortho  
export ZOOM=${ZOOM:-$zoom}


export Camera=INFO

export NAMEPREFIX=cxr_geochain_${GEOM}_   # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_geochain/cam_${ICAM}
export TOPLINE="./cxr_geochain.sh $MOI      # EYE $EYE  $stamp  $version $GEOM   " 

vars="GEOM MOI CFNAME EMM TMIN EYE ICAM NAMEPREFIX RELDIR TOPLINE"
echo $msg 
for var in $vars ; do printf "%-20s : %s \n" $var "${!var}" ; done  

echo $msg invoke ./cxr.sh 
./cxr.sh 

exit 0

