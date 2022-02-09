#!/bin/bash -l

usage(){ cat << EOU
GeneralSphereDEV.sh
=====================

After making changes here 

Visualize Geant4 polyhedron in 3D::

   ./X4MeshTest.sh 

Visualize Geant4 intersects in 2D::

   ./xxs.sh 

Updating CSGFoundry geometry using GeoChain, enables the below tests::

    gc   # cd ~/opticks/GeoChain
    GEOM=GeneralSphereDEV ./run.sh 

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


#export X4SolidMaker_GeneralSphereDEV_innerRadius=50
export X4SolidMaker_GeneralSphereDEV_innerRadius=0
export X4SolidMaker_GeneralSphereDEV_outerRadius=100

# two azimuthal phi angles (start, start+delta) should be in range 0.->2. 
export X4SolidMaker_GeneralSphereDEV_phiStart=0.0
export X4SolidMaker_GeneralSphereDEV_phiDelta=2.0



# two polar theta angles ( start, start+delta ) should be in range 0.->1. 

#thetaMode=full
#thetaMode=tophole
#thetaMode=topdiamond
#thetaMode=bothole
thetaMode=butterfly
#thetaMode=belt

case $thetaMode in 
        full)    thetaStart=0.00 ; thetaDelta=1.00 ;; 
     tophole)    thetaStart=0.10 ; thetaDelta=0.90 ;; 
     topdiamond) thetaStart=0.10 ; thetaDelta=0.10 ;; 
     bothole)    thetaStart=0.00 ; thetaDelta=0.90 ;; 
   butterfly)    thetaStart=0.25 ; thetaDelta=0.50 ;;
        belt)    thetaStart=0.45 ; thetaDelta=0.10 ;; 
esac

export X4SolidMaker_GeneralSphereDEV_thetaStart=$thetaStart
export X4SolidMaker_GeneralSphereDEV_thetaDelta=$thetaDelta

echo $BASH_SOURCE thetaMode $thetaMode
env | grep X4SolidMaker_GeneralSphereDEV

