#!/bin/bash -l

opticks-
om-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir 

cd $bdir 
pwd 
om-cmake $sdir
make
make install   



thoughts(){ cat << EOT

Finding Geant4
=================

Note that Geant4 is found via:: 

  -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 

There is no need (so long as only one G4 version in externals) to use::

  -DGeant4_DIR=$(g4-cmake-dir)

Actually with multiple versions of Geant4 in externals CMake finds
the latest one.

EOT
}


