#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

## Opticks external Geant4 found via CMAKE_PREFIX_PATH

make
[ "$(uname)" == "Darwin" ] && echo "Kludge sleep 2s" && sleep 2 
make install   


bin=$(opticks-prefix)/lib/$name
ls -l $bin
$bin


