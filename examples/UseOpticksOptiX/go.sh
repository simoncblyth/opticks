#!/bin/bash -l
opticks-

sdir=$(pwd)

name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 

om-
om-cmake $sdir

#make
#make install   




