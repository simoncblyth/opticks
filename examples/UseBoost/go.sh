#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

make
make install   

