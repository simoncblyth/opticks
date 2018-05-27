#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)

#bdir=/tmp/$USER/opticks/$name/build 
bdir=$(opticks-prefix)/build/$name

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     \
    -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
    -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)



notes(){ cat << EOT

Finding Externals
===================

OptiX
    OptiX_INSTALL_DIR is required


Geant4
    via -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 
    no need (so long as only one G4 version in externals) to use::
    
       -DGeant4_DIR=$(g4-cmake-dir)

EOT
}


make

if [ "$(uname)" == "Darwin" ]; then
   echo "kludge sleeping for 2s"
   sleep 2
fi 

make install   


$(opticks-prefix)/lib/OKConfTest


bcm_deploy_conf=$(opticks-prefix)/lib/cmake/okconf/okconf-config.cmake
ls -l $bcm_deploy_conf
cat $bcm_deploy_conf



