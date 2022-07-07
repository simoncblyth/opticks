#!/bin/bash -l 
usage(){ cat << EOU
GEOM_.sh
===========

This script is sourced from the below scripts to 
define the GEOM envvar for setup of test geometries. 

u4/tests/U4RecorderTest.sh 
gx/gxs.sh 


EOU
}

#geom=BoxOfScintillator
#geom=RaindropRockAirWater
#geom=RaindropRockAirWaterSD
geom=hama_body_log

export GEOM=${GEOM:-$geom}

if [ "$GEOM" == "RaindropRockAirWater" -o "$GEOM" == "RaindropRockAirWaterSD" ]; then 
    export U4VolumeMaker_RaindropRockAirWater_FACTOR=10
fi 


echo === $BASH_SOURCE : GEOM $GEOM 

