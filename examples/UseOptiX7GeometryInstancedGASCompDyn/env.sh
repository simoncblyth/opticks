#!/bin/bash 

sdir=$(pwd)
name=$(basename $sdir)

export PREFIX=/tmp/$USER/opticks/$name
export PATH=$PREFIX/bin:$PATH
export BIN=$(which $name)

#tmin=2.0
tmin=0.5
#tmin=0.1

#geometry=sphere
#geometry=sphere_two
#geometry=sphere_containing_grid_of_two_radii_spheres
geometry=sphere_containing_grid_of_two_radii_spheres_compound

#gridspec=-10:11:2,-10:11:2,-10:11:2
#gridspec=-40:41:4,-40:41:4,-40:41:4
gridspec=-40:41:10,-40:41:10,-40:41:10
#gridspec=-40:41:10,-40:41:10,0:1:1

#eye=-0.5,-0.5,0.5
eye=-1.0,-1.0,1.0

fudge=1
#fudge=2

cameratype=0


modulo=0,1
#single=2
single=""


# make sensitive to calling environment
export GEOMETRY=${GEOMETRY:-$geometry}
export FUDGE=${FUDGE:-$fudge}
export TMIN=${TMIN:-$tmin}
export CAMERATYPE=${CAMERATYPE:-$cameratype}
export GRIDSPEC=${GRIDSPEC:-$gridspec}
export EYE=${EYE:-$eye} 
export MODULO=${MODULO:-$modulo}
export SINGLE=${SINGLE:-$single}


export OUTDIR=$PREFIX/$GEOMETRY/FUDGE_${FUDGE}_TMIN_${TMIN}

echo name       : $name
echo BIN        : $BIN
echo FUDGE      : $FUDGE
echo TMIN       : $TMIN
echo CAMERATYPE : $CAMERATYPE
echo GRIDSPEC   : $GRIDSPEC
echo EYE        : $EYE

echo PREFIX   : $PREFIX
echo GEOMETRY : $GEOMETRY
echo OUTDIR   : $OUTDIR


