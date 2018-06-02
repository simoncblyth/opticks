#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

if [ "$1" == "clean" ]; then
    rm -rf $bdir 
fi 

if [ "$1" == "conf" ]; then
    mkdir -p $bdir && cd $bdir && pwd 
    cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
                -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
                -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
                -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

else
    cd $bdir
fi 


make
make install   


$(opticks-prefix)/lib/UseYoctoGLRap

