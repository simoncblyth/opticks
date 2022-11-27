#!/bin/bash -l 
usage(){ cat << EOU
GEOM_.sh
===========

Edit this script using "geom_" bash function from opticks/opticks.bash 

* NB this GEOM_.sh script is distinct from the "geom" bash function and GEOM.txt file
  which does similar, but traditionally has been used with very small single solid geometries
  
* TODO: consolidate the two geometry config approaches


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

NOTE that this does not set CFBASE but as that was found 
confusing as sometimes an executables loads from CFBASE and sometimes
it writes to it. 

EOU
}

#geom=J001
#geom=J003
#geom=J004

#geom=BoxOfScintillator
#geom=RaindropRockAirWater
#geom=RaindropRockAirWaterSD
#geom=RaindropRockAirWaterSmall

#geom=hamaBodyLog

#geom=nnvtPMTSolid
#geom=nnvtBodySolid
#geom=nnvtInner1Solid
#geom=nnvtInner2Solid

#geom=hamaPMTSolid
#geom=hamaDynodeSolid   ## seems dynode changed, needs reworking in PMTSim

#geom=hmskSolidMaskVirtual
#geom=hmskSolidMask
#geom=hmskSolidMaskTail
#geom=hamaPMTSolid
#geom=hamaBodySolid
#geom=hamaInner1Solid
#geom=hamaInner2Solid

geom=hamaLogicalPMTWrapLV


#
#geom=hmskTailOuterIEllipsoid
#geom=hmskTailOuterITube             # hz 0.15 cylinder 
#geom=hmskTailOuterI                 # union of hmskTailOuterIEllipsoid and hmskTailOuterITube 
#geom=hmskTailOuterIITube            # hz ~72 cylinder
#geom=hmskTailOuter                   # union of above union and the big cylidner  
#
#geom=hmskTailInnerIEllipsoid
#geom=hmskTailInnerITube
#geom=hmskTailInnerI
#geom=hmskTailInnerIITube
#geom=hmskTailInner


#geom=nmskSolidMask
#geom=nmskSolidMaskTail

#geom=nmskSolidMaskTail     ## NB no need for __U1 opt here, that is set below
#geom=nmskSolidMaskVirtual

#geom=nmskTailOuter
#geom=nmskTailInner
#geom=nmskMaskOut
#geom=nmskMaskIn
#geom=nmskTailOuterIEllipsoid
#geom=nmskTailOuterITube
#geom=nmskTailOuterI
#geom=nmskTailOuterIITube

#geom=nmskTailInnerIEllipsoid
#geom=nmskTailInnerITube
#geom=nmskTailInnerI
#geom=nmskTailInnerIITube 

#geom=acyl
#geom=cyli

#opt=U1BUG33  # U2 
opt=U1
case $geom in 
   nmsk*|hmsk*|nnvt*|hama*) geom=${geom}__${opt} ;;
esac

export GEOM=${GEOM:-$geom}  ## NB pre-existing GEOM overrides all the above 


# transitional : from old OPTICKS_KEY geometry dirs 
reldir(){
   case $1 in 
     J0*) echo DetSim0Svc_pWorld_g4live/g4ok_gltf/41c046fe05b28cb70b1fc65d0e6b7749/1 ;;
   esac
}

if [ "$GEOM" == "J000" ]; then

    export J000_CFBaseFromGEOM=$HOME/.opticks/geocache/$(reldir $GEOM)/CSG_GGeo
    ## HMM: cannot u4s.sh with this as no GDMLPath

elif [ "$GEOM" == "J001" ]; then

    ## starts from GDML and does translation when no  _CFBaseFromGEOM export 
    export J001_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml 

    #export GInstancer_instance_repeat_min=1000000  
    # default is 400, setting to very high value will make everything global 

elif [ "$GEOM" == "J002" ]; then

    export J002_GDMLPath=$HOME/.opticks/geocache/$(reldir $GEOM)/origin_CGDMLKludge.gdml 
    export J002_GEOMSub=HamamatsuR12860sMask_virtual0x:0:1000
    export J002_GEOMWrap=AroundSphere 

elif [ "$GEOM" == "ntds3" ]; then 

    export ntds3_GDMLPathFromGEOM=/tmp/$USER/opticks/GEOM/$GEOM/G4CXOpticks/origin.gdml

elif [ "$GEOM" == "J003" ]; then

    export J003_CFBaseFromGEOM=$HOME/.opticks/ntds3/G4CXOpticks

elif [ "$GEOM" == "J004" ]; then

    # from jxf: save the geometry from junosw using  "GEOM=J004 ntds3" 
    # this uses envvar G4CXOpticks__setGeometry_saveGeometry to signal the save and pass the directory 
    cfbase=$HOME/.opticks/GEOM/$GEOM
    export J004_CFBaseFromGEOM=$cfbase
    export J004_GDMLPathFromGEOM=$cfbase/origin.gdml

elif [ "$GEOM" == "J004G" ]; then

    export J004G_GDMLPath=$HOME/.opticks/GEOM/J004/origin.gdml

elif [ "$GEOM" == "example_pet" ]; then

    export example_pet_GDMLPathFromGEOM=$HOME/geant_pet/example_pet_opticks/small_pet.gdml

else 
    # handling test geometries from j/PMTSim aka jps and from GeoChain or CSGMakerTest 
    export ${GEOM}_CFBaseFromGEOM=/tmp/$USER/opticks/GEOM/$GEOM 
    #export ${GEOM}_CFBaseFromGEOM=$HOME/.opticks/GEOM/$GEOM
fi 


##### TODO: GET RID OF THE BELOW : GEOM SHOULD FOCUS ON VITALS #######


case $GEOM in 
 RaindropRockAirWaterSmall) export U4VolumeMaker_RaindropRockAirWater_FACTOR=1 ;;
      RaindropRockAirWater) export U4VolumeMaker_RaindropRockAirWater_FACTOR=10 ;;
    RaindropRockAirWaterSD) export U4VolumeMaker_RaindropRockAirWater_FACTOR=10 ;;
esac
## HMM: move this kinda specifics to geoscript perhaps



gp_=${GEOM}_GDMLPath 
gp=${!gp_}
cg_=${GEOM}_CFBaseFromGEOM
cg=${!cg_}

# CFBASE is the directory that contains (or will contain) the CSGFoundry geometry folder 

TMP_GEOMDIR=/tmp/$USER/opticks/GEOM/$GEOM
GEOMDIR=${cg:-$TMP_GEOMDIR}
export GEOMDIR 

if [ -z "$QUIET" ]; then 
   vars="BASH_SOURCE gp_ gp cg_ cg TMP_GEOMDIR GEOMDIR BASH_SOURCE" 
   for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done  
   echo 
fi 

upfind_cfbase(){
    : opticks/bin/GEOM_.sh : traverse directory tree upwards searching the CFBase geometry dir identified by existance of relative CSGFoundry/solid.npy  
    local dir=$1
    while [ ${#dir} -gt 1 -a ! -f "$dir/CSGFoundry/solid.npy" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}



