#!/bin/bash
usage(){ cat << EOU
G4CX_U4TreeCreateCSGFoundryTest.sh
===================================

~/o/g4cx/tests/G4CX_U4TreeCreateCSGFoundryTest.sh

Creates Geant4 PV configured with GEOM envvar,
converts to Opticks stree/CSGFoundry and persists
the CSGFoundry into ~/.opticks/GEOM.

Visualize the result with ~/o/cxr_min.sh

For a lower level check of the solid use::

   ~/o/u4/tests/U4SolidMakerTest.sh


EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=G4CX_U4TreeCreateCSGFoundryTest

source $HOME/.opticks/GEOM/GEOM.sh
fold=$HOME/.opticks/GEOM/$GEOM
export FOLD=$fold




if [ "$GEOM" == "LocalOuterReflectorOrbSubtraction" ]; then
    rad=$(echo 20722.1 + 20 | bc)   ## despite triangualated big Orb 20mm is enough to make subtractions contained
    [ $? -ne 0 ] && echo $BASH_SOURCE ERROR && exit 1
    #export U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector=$rad
    unset U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector

    #export U4Mesh__NumberOfRotationSteps_entityType_G4Orb=480  ## NOPE NAME OF ROOT SOLID NEEDED
    export U4Mesh__NumberOfRotationSteps_solidName_OuterReflectorOrbSubtraction_cutTube2=480

    #export U4SolidMaker__MakeLowerWaterDistributorCurvedCutTubes_UNCOINCIDE_MM=-1  # -1 : splits the tubes
    #export U4SolidMaker__MakeLowerWaterDistributorCurvedCutTubes_UNCOINCIDE_MM=1    # 1 : succeeds to uncoincide
    export U4SolidMaker__MakeLowerWaterDistributorCurvedCutTubes_UNCOINCIDE_MM=0    # 0 : default coincident face

elif [[ "$GEOM" =~ ^LocalR12860_PMTSolid[UK]?$ ]]; then

    export U4Mesh__NumberOfRotationSteps_solidName_${GEOM/Local}_pmt_solid=480

fi


defarg="info_run"
arg=${1:-$defarg}

vv="BASH_SOURCE name GEOM fold FOLD defarg arg"
vv="$vv U4SolidMaker__OuterReflectorOrbSubtraction_radOuterReflector"


if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/run}" != "$arg" -a "$YES" == "1" ]; then

   $name
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1

elif [ "${arg/run}" != "$arg" ]; then

   if [ -d "$FOLD" ]; then
      ans="NO"
      echo $BASH_SOURCE FOLD $FOLD exists already
      read -p "$BASH_SOURCE - Enter YES to proceed to overwrite into this FOLD : " ans
   else
      ans="YES"
   fi

   if [ "$ans" == "YES" ]; then
      $name
      [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 1
   else
      echo $BASH_SOURCE - skipping
   fi
fi


exit 0


