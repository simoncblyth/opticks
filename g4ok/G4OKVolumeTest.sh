#!/bin/bash -l 
usage(){ cat << EOU
G4OKVolumeTest.sh : creates G4VPhysicalVolume and converts it into Opticks geocache
======================================================================================

Volumes are created by X4VolumeMaker::Make, see extg4/X4VolumeMakerTest.sh for GEOM names 
Some envvars allow some control over the conversion:

GInstancer_instance_repeat_min
   large number (or comment it giving default of 400) creates an all global geometry
   small number does more instancing 

Usage::

   g4ok
   GEOM=SomeName ./G4OKVolumeTest.sh 

What g4ok/tests/G4OKVolumeTest.cc does: 

1. Creates Geant4 volume tree and converts it into an Opticks/GGeo
   geocache which is saved. 

2. A script with the OPTICKS_KEY config is written to a standard location, 
   that can be used by opticks-hookup $OPTICKS_HOME/bin/geocache_hookup.sh 
   when using the "last" argument. 


To visualize the "last" created geometry::

   ok
   ./ott.sh    # default arg is "last"

To convert the "last" created geometry to CSGFoundry::

   cg ;     cd ~/opticks/CSG_GGeo
   # ./run.sh     # run.sh is giving error from lack of tran.npy  in CSGFoundry 
   ./rundbg.sh    # to dump the GParts and examine them with python : 


EOU
}

msg="=== $BASH_SOURCE :"


#geom=JustOrbGrid
#geom=JustOrbCube
#geom=BoxMinusOrbCube
#geom=lchilogicLowerChimney
geom=ListJustOrb,BoxMinusOrb
#geom=BoxMinusOrbXoff
#geom=JustOrbZoff

export GEOM=${GEOM:-$geom}

export X4PhysicalVolume=INFO
export X4VolumeMaker=INFO
export GParts=INFO
export GInstancer=INFO


export GInstancer_instance_repeat_min=5000
echo $msg GInstancer_instance_repeat_min $GInstancer_instance_repeat_min

## NB recall that Opticks is embedded inside G4Opticks so cannot directly pass commandline options to it 

if [ -n "$DEBUG" ]; then
   lldb__ G4OKVolumeTest
else
    G4OKVolumeTest 
fi 


