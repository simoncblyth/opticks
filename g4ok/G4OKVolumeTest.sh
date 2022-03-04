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


   cd ~/opticks/CSG_GGeo

   # ./run.sh     # run.sh is giving error from lack of tran.npy  in CSGFoundry 

   ./rundbg.sh    # to dump the GParts and examine them with python : 


EOU
}

msg="=== $BASH_SOURCE :"

#geom=JustOrbGrid
#geom=JustOrbCube
geom=BoxMinusOrbCube


export GEOM=${GEOM:-$geom}

export X4PhysicalVolume=INFO
export GParts=INFO
export GInstancer=INFO

# comment the below to create an all global geometry, uncomment to instance the PMT volume 
#export GInstancer_instance_repeat_min=25


## NB recall that Opticks is embedded to cannot directly pass commandline options to it 

if [ -n "$DEBUG" ]; then
   lldb__ G4OKVolumeTest
else
    G4OKVolumeTest 

    
fi 






