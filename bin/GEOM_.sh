#!/bin/bash -l 
usage(){ cat << EOU
GEOM_.sh
===========

This script is sourced from the below scripts to 
define the GEOM envvar for setup of test geometries. 

u4/tests/U4RecorderTest.sh 
gx/gxs.sh 

Test with::

   source ~/opticks/bin/GEOM_.sh  
   SOpticksResourceTest 
   CSGFoundry_ResolveCFBase_Test 
   exit   # the dirty shell 


EOU
}

#geom=BoxOfScintillator
#geom=RaindropRockAirWater
#geom=RaindropRockAirWaterSD
#geom=hama_body_log
geom=J000

export GEOM=${GEOM:-$geom}

reldir(){
   case $1 in 
     J000) echo DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1 ;;
   esac
}

if [ "$GEOM" == "J000" ]; then
    export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml 
    export J000_CFBaseFromGEOM=$HOME/.opticks/geocache/$(reldir $GEOM)/CSG_GGeo
    ## to force translation from GDML comment the _CFBaseFromGEOM export 
fi 

case $GEOM in 
    RaindropRockAirWater|RaindropRockAirWaterSD) export U4VolumeMaker_RaindropRockAirWater_FACTOR=10 ;;
esac



gp_=${GEOM}_GDMLPath 
gp=${!gp_}
cg_=${GEOM}_CFBaseFromGEOM
cg=${!cg_}

if [ -n "$cg" ]; then
    CFBASE=$cg 
    A_FOLD=$CFBASE/G4CXSimulateTest/ALL
else
    CFBASE=/tmp/$USER/opticks/G4CXSimulateTest/$GEOM
    A_FOLD=$CFBASE/ALL
fi
# NB CFBASE is NOT exported here : it is exported for the python ana, not the C++ run 

if [ -z "$QUIET" ]; then 
   echo === $BASH_SOURCE : GEOM $GEOM CFBASE $CFBASE 
fi 

