#!/bin/bash -l

usage(){ cat << EOU
GeneralSphereDEV.sh
=====================

After making changes to envvars here can immediately 
do Geant4 level visualizations

Geant4 Visualizations
-------------------------

Visualize Geant4 polyhedron in 3D::

   ./X4MeshTest.sh 

Visualize Geant4 intersects in 2D::

   ./xxs.sh 


Convert Geant4 Geometry into OpticksCSG
------------------------------------------

Updating CSGFoundry geometry using GeoChain, enables the Opticks visualizations::

    gc   # cd ~/opticks/GeoChain
    GEOM=GeneralSphereDEV ./run.sh 


Opticks Visualizations
------------------------

2D CPU intersect::

    c
    ./csg_geochain.sh 

2D GPU intersect::

    cx
    ./cxs_geochain.sh 

3D GPU render::

    cx
    ./cxr_geochain.sh 

EOU
}

############### two radii 

radiusMode=full
#radiusMode=half
case $radiusMode in 
    full) innerRadius=0  ; outerRadius=100 ;;
    half) innerRadius=50 ; outerRadius=100 ;; 
esac  

export X4SolidMaker_GeneralSphereDEV_radiusMode=$radiusMode
export X4SolidMaker_GeneralSphereDEV_innerRadius=$innerRadius
export X4SolidMaker_GeneralSphereDEV_outerRadius=$outerRadius 

######### two azimuthal phi angles (start, start+delta) should be in range 0.->2.  #########

#phiMode=full
#phiMode=melon
#phiMode=pacman
phiMode=pacmanpp

case $phiMode in 
        full)    phiStart=0.00 ; phiDelta=2.00 ;; 
       melon)    phiStart=0.25 ; phiDelta=0.50 ;;   
      pacman)    phiStart=0.25 ; phiDelta=1.50 ;;   
      pacmanpp)  phiStart=0.50 ; phiDelta=1.50 ;;   
esac
export X4SolidMaker_GeneralSphereDEV_phiMode=$phiMode
export X4SolidMaker_GeneralSphereDEV_phiStart=$phiStart
export X4SolidMaker_GeneralSphereDEV_phiDelta=$phiDelta


######## two polar theta angles ( start, start+delta ) should be in range 0.->1. ##########

thetaMode=full
#thetaMode=tophole
#thetaMode=topdiamond
#thetaMode=bothole
#thetaMode=butterfly
#thetaMode=belt

case $thetaMode in 
        full)    thetaStart=0.00 ; thetaDelta=1.00 ;; 
     tophole)    thetaStart=0.10 ; thetaDelta=0.90 ;; 
     topdiamond) thetaStart=0.10 ; thetaDelta=0.10 ;; 
     bothole)    thetaStart=0.00 ; thetaDelta=0.90 ;; 
   butterfly)    thetaStart=0.25 ; thetaDelta=0.50 ;;
        belt)    thetaStart=0.45 ; thetaDelta=0.10 ;; 
esac

export X4SolidMaker_GeneralSphereDEV_thetaMode=$thetaMode
export X4SolidMaker_GeneralSphereDEV_thetaStart=$thetaStart
export X4SolidMaker_GeneralSphereDEV_thetaDelta=$thetaDelta

########

echo $BASH_SOURCE 
env | grep X4SolidMaker_GeneralSphereDEV

