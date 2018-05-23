#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/build
#bdir=$(opticks-prefix)/build

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
    -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir)

make

if [ "$(uname)" == "Darwin" ]; then
   echo "kludge sleeping for 2s"
   sleep 2
fi 

make install   

