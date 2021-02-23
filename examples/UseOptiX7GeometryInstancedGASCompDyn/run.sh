#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir)


prefix=/tmp/$USER/opticks/$name

rm -rf $prefix/ppm
rm -rf $prefix/npy
mkdir -p $prefix/{ppm,npy} 


export PREFIX=$prefix
export PATH=$PREFIX/bin:$PATH

bin=$(which $name)
spec=$1

#export GEOMETRY=sphere_two
#export GEOMETRY=sphere_containing_grid_of_two_radii_spheres_compound
export GEOMETRY=sphere_containing_grid_of_two_radii_spheres

#gdb -ex r --args $bin $spec
$bin $spec

[ $? -ne 0 ] && echo $0 : run  FAIL && exit 3

ppm=$prefix/ppm/$name.ppm

echo name : $name
echo bin  : $(which $name)
echo spec : $spec
echo ppm  : $ppm
echo md5  : $(cat $ppm | md5sum)

exit 0

