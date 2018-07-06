#!/bin/bash -l

opticks-
xercesc-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 


#            -DXERCESC_LIBRARY=$(xercesc-library) \
#            -DXERCESC_INCLUDE_DIR=$(xercesc-include-dir)

make
make install   

