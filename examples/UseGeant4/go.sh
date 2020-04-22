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
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

Finding Geant4
=================

Note that Geant4 is found via:: 

  -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 

There is no need (so long as only one G4 version in externals) to use::

  -DGeant4_DIR=$(g4-cmake-dir)

  -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals 



EOT
}

om-
om-export
om-export-info

pkg=Geant4
find_package.py $pkg
pkg_config.py $pkg #--level debug


  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

make
make install   


UseGeant4

