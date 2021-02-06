#!/bin/bash -l

opticks-
om-

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

om-cmake $sdir 
make
make install   

$name



