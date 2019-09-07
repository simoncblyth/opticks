#!/bin/bash -l
##
## Copyright (c) 2019 Opticks Team. All Rights Reserved.
##
## This file is part of Opticks
## (see https://bitbucket.org/simoncblyth/opticks).
##
## Licensed under the Apache License, Version 2.0 (the "License"); 
## you may not use this file except in compliance with the License.  
## You may obtain a copy of the License at
##
##   http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software 
## distributed under the License is distributed on an "AS IS" BASIS, 
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
## See the License for the specific language governing permissions and 
## limitations under the License.
##


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



