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

opticks-
oe-
om-

echo $CMAKE_PREFIX_PATH | tr ":" "\n"

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

om-cmake $sdir 
make
make install   

echo $BASH_SOURCE : running executable $name 

$name
rc=$?

echo $BASH_SOURCE : $name executable rc $rc 

