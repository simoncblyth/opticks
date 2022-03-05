#!/bin/bash -l 

usage(){ cat << EOU

G4OKVolumeTest.sh : creates G4VPhysicalVolume and converts it into Opticks geocache
======================================================================================

* volumes are created by X4VolumeMaker::Make

Usage::

   g4ok
   GEOM=SomeName ./G4OKVolumeTest.sh 

As these geocache are generally throwaways whilst testing can just 
inline use the OPTICKS_KEY to check visualization and CSGFoundry conversion.

   cd ~/opticks/CSG_GGeo

   # ./run.sh     # run.sh is giving error from lack of tran.npy  in CSGFoundry 

   ./rundbg.sh    # to dump the GParts and examine them with python : 


GInstancer_instance_repeat_min
   large number (or comment it giving default of 400) creates an all global geometry
   small number does more instancing 


EOU
}

msg="=== $BASH_SOURCE :"

#geom=JustOrbGrid
#geom=JustOrbCube
#geom=BoxMinusOrbCube
#geom=lchilogicLowerChimney

geom=ListJustOrb,BoxMinusOrb

export GEOM=${GEOM:-$geom}

export X4PhysicalVolume=INFO
export GParts=INFO
export GInstancer=INFO


export GInstancer_instance_repeat_min=5000

echo $msg GInstancer_instance_repeat_min $GInstancer_instance_repeat_min



## NB recall that Opticks is embedded to cannot directly pass commandline options to it 

if [ -n "$DEBUG" ]; then
   lldb__ G4OKVolumeTest
else
    G4OKVolumeTest 
fi 





