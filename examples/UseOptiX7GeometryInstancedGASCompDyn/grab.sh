#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir)
prefix=/tmp/$USER/opticks/$name


#export GEOMETRY=sphere
#export GEOMETRY=sphere_two
#export GEOMETRY=sphere_containing_grid_of_two_radii_spheres
export GEOMETRY=sphere_containing_grid_of_two_radii_spheres_compound

dir=$prefix/$GEOMETRY
mkdir -p $dir

ppm=$dir/pixels.ppm
npy=$dir/posi.npy

ppm_()
{
   echo scp P:$ppm $ppm
   scp P:$ppm $ppm
   open $ppm 
}

npy_()
{
   echo scp P:$npy $npy
   scp P:$npy $npy
   ipython -i posi.py  
}


if [ "$1" == "ppm" ]; then
   ppm_
elif [ "$1" == "npy" ]; then
   npy_
else
   ppm_
   npy_
fi 




