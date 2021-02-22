#!/bin/bash -l

opticks-
om-

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/opticks/$name/build 
rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

om-cmake $sdir
make
make install   

exe=$(opticks-prefix)/lib/$name

echo "running exe $exe"
eval $exe


