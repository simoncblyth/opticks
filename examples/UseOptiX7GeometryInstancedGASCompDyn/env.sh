#!/bin/bash 

sdir=$(pwd)
name=$(basename $sdir)

export PREFIX=/tmp/$USER/opticks/$name
export PATH=$PREFIX/bin:$PATH
export BIN=$(which $name)


#export GEOMETRY=sphere
#export GEOMETRY=sphere_two
#export GEOMETRY=sphere_containing_grid_of_two_radii_spheres
export GEOMETRY=sphere_containing_grid_of_two_radii_spheres_compound

#export FUDGE=2
#export TMIN=2

export OUTDIR=$PREFIX/$GEOMETRY

echo name     : $name
echo BIN      : $BIN
echo PREFIX   : $PREFIX
echo GEOMETRY : $GEOMETRY
echo OUTDIR   : $OUTDIR

rm -rf $OUTDIR
mkdir -p $OUTDIR


