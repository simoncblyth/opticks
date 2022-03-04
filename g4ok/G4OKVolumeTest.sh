#!/bin/bash -l 

usage(){ cat << EOU

G4OKVolumeTest.sh : creates G4VPhysicalVolume and converts it into Opticks geocache
======================================================================================

* volumes are created by X4VolumeMaker::Make

Usage::

   g4ok
   GEOM=SomeName ./G4OKVolumeTest.sh 
   OPTICKS_KEY=G4OKVolumeTest......  OTracerTest 

As these geocache are generally throwaways whilst testing can just 
inline use the OPTICKS_KEY to check visualization and CSGFoundry conversion.

EOU
}

#geom=JustOrbGrid
geom=JustOrbCube


export GEOM=${GEOM:-$geom}

export X4PhysicalVolume=INFO
export GInstancer=INFO

# comment the below to create an all global geometry, uncomment to instance the PMT volume 
#export GInstancer_instance_repeat_min=25

if [ -n "$DEBUG" ]; then
   lldb__ G4OKVolumeTest
else
    G4OKVolumeTest
fi 


