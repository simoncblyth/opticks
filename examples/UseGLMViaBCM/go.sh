#!/bin/bash -l

usage(){ cat << EOU

Simple script to exercise CMake machinery, 
running this script does:

* cmake configuration, generating Makefiles into bdir beneath /tmp
* building 
* installing into opticks-prefix, so ensure the basename of the directory is unique
* run the resulting executable

Everything is done on every invokation.


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

# CONFIG mode find_package rules allow the BCM machinery in  <prefix>/share/bcm/cmake/ 
# to be found without assistance (ie no need to modify CMAKE_MODULE_PATH)

make
make install   



