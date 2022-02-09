#!/bin/bash -l

export X4SolidMaker_GeneralSphereDEV_innerRadius=50
export X4SolidMaker_GeneralSphereDEV_outerRadius=100

# two azimuthal phi angles (start, start+delta) should be in range 0.->2. 
export X4SolidMaker_GeneralSphereDEV_phiStart=0.0
export X4SolidMaker_GeneralSphereDEV_phiDelta=2.0

# two polar theta angles ( start, start+delta ) should be in range 0.->1. 
export X4SolidMaker_GeneralSphereDEV_thetaStart=0.25
export X4SolidMaker_GeneralSphereDEV_thetaDelta=0.50

echo $BASH_SOURCE
env | grep X4SolidMaker_GeneralSphereDEV

