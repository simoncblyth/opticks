#!/bin/bash -l 

export X4VolumeMaker=INFO

 
#geom=JustOrbGrid
#geom=JustOrbCube
#geom=ListJustOrb,BoxMinusOrb
#geom=JustOrbXoff
geom=BoxMinusOrbXoff


export GEOM=${GEOM:-$geom}

X4VolumeMakerTest



