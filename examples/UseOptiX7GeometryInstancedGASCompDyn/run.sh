#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir)

usage(){ cat << EOU

::

    FUDGE=2 TMIN=2 ./run.sh

TODO:

* add posi integer bitfeld identifying : instance_idx, gas_idx, bi_idx (within the gas), sbt_idx?
* somehow communicate the geo parameters to the python so can calc the sdf of all intersects

EOU
}

prefix=/tmp/$USER/opticks/$name

rm -rf $prefix/ppm
rm -rf $prefix/npy
mkdir -p $prefix/{ppm,npy} 

export PREFIX=$prefix
export PATH=$PREFIX/bin:$PATH

bin=$(which $name)
spec=$1

#export GEOMETRY=sphere
#export GEOMETRY=sphere_two
#export GEOMETRY=sphere_containing_grid_of_two_radii_spheres
export GEOMETRY=sphere_containing_grid_of_two_radii_spheres_compound

#export FUDGE=2
#export TMIN=2

#gdb -ex r --args $bin $spec
$bin $spec

[ $? -ne 0 ] && echo $0 : run  FAIL && exit 3

ppm=$prefix/ppm/$name.ppm
npy=$prefix/npy/$name.npy

echo name : $name
echo bin  : $(which $name)
echo spec : $spec
echo ppm  : $ppm
echo md5  : $(cat $ppm | md5sum)
echo npy  : $npy

exit 0

