#!/bin/bash -l 
usage(){ cat << EOU
GEOM_.sh
===========

This script is sourced from the below scripts to 
define the GEOM envvar for setup of test geometries. 

u4/u4s.sh 
gx/gxs.sh 

Test with::

   source ~/opticks/bin/GEOM_.sh  
   SOpticksResourceTest 
   CSGFoundry_ResolveCFBase_Test 
   exit   # the dirty shell 


See notes with SOpticksResource::DefaultOutputDir for discussion of change to directory layout. 

EOU
}

#geom=BoxOfScintillator

#geom=RaindropRockAirWater
#geom=RaindropRockAirWaterSD
#geom=RaindropRockAirWaterSmall

#geom=hama_body_log
geom=J000

export GEOM=${GEOM:-$geom}

reldir(){
   case $1 in 
     J000) echo DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1 ;;
   esac
}

if [ "$GEOM" == "J000" ]; then

    export J000_CFBaseFromGEOM=$HOME/.opticks/geocache/$(reldir $GEOM)/CSG_GGeo

elif [ "$GEOM" == "J001" ]; then

    ## starts from GDML and does translation when no  _CFBaseFromGEOM export 
    export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml 

    #export GInstancer_instance_repeat_min=1000000  
    # default is 400, setting to very high value will make everything global 

elif [ "$GEOM" == "J002" ]; then

    export J000_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml 
    export J000_GEOMSub=HamamatsuR12860sMask_virtual0x:0:1000
    export J000_GEOMWrap=AroundSphere 

fi 

case $GEOM in 
 RaindropRockAirWaterSmall) export U4VolumeMaker_RaindropRockAirWater_FACTOR=1 ;;
      RaindropRockAirWater) export U4VolumeMaker_RaindropRockAirWater_FACTOR=10 ;;
    RaindropRockAirWaterSD) export U4VolumeMaker_RaindropRockAirWater_FACTOR=10 ;;
esac



gp_=${GEOM}_GDMLPath 
gp=${!gp_}
cg_=${GEOM}_CFBaseFromGEOM
cg=${!cg_}

# CFBASE is the directory that contains (or will contain) the CSGFoundry geometry folder 

TMP_GEOMDIR=/tmp/$USER/opticks/$GEOM
GEOMDIR=${cg:-$TMP_GEOMDIR}

export GEOMDIR 

if [ -z "$QUIET" ]; then 
   vars="BASH_SOURCE TMP_GEOMDIR GEOMDIR" 
   for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done  
   echo 
fi 

notes(){ cat << EON

Formerly this set CFBASE but thats confusing 
as sometimes an executables loads from CFBASE and sometimes
it writes to it. 

EON
}


