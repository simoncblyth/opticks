#!/bin/bash -l
usage(){ cat << EOU
UseCustom4/go.sh 
==================

Testing CMake find_package with Custom4

To remove all installed Custom4 libs and headers::

   rm -rf /usr/local/opticks/lib/Custom4*
   rm -rf /usr/local/opticks/include/Custom4

EOU
}

echo $CMAKE_PREFIX_PATH | tr ":" "\n"

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 
idir=/tmp/$USER/opticks/$name/install

rm -rf $idir && mkdir -p $idir && cd $idir && pwd 
rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$idir 
make
make install   

bin=$idir/lib/$name

echo $BASH_SOURCE : bin $bin : running 

$bin
rc=$?

echo $BASH_SOURCE : bin $bin : rc $rc 

