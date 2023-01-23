#!/bin/bash -l 

usage(){ cat << EOU
C4Test.sh 
===========

EOU
}

loglevels(){
   export U4VolumeMaker=INFO
}

loglevels

#export GEOM=RaindropRockAirWater
export GEOM=J006
export J006_GDMLPath=$HOME/.opticks/GEOM/$GEOM/origin.gdml

export SSim__stree_level=1 

C4Test



