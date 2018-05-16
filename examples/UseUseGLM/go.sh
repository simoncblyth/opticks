#!/bin/bash -l

usage(){ cat << EOU


EOU
}


opticks-

home=$(opticks-home)
prefix=$(opticks-prefix)

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 
idir=/tmp/$USER/$name/install

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_MODULE_PATH=$home/cmake/Modules

make
make install   

exe=$prefix/lib/$name


if [ -f "$exe" ]; then
    ls -l $exe
    echo running installed exe $exe
    $exe
else 
    echo failed to install exe to $exe 
fi 



