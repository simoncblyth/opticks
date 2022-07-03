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
geom=RaindropRockAirWater2
#geom=hama_body_log

export GEOM=${GEOM:-$geom}

echo === $BASH_SOURCE : GEOM $GEOM 

