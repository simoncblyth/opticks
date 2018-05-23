#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

#rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

Finding Geant4
=================

Note that Geant4 is found via:: 

  -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 

There is no need (so long as only one G4 version in externals) to use::

  -DGeant4_DIR=$(g4-cmake-dir)

EOT
}
  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

make
make install   

