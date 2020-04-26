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
oe-
om-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

echo bdir $bdir name $name

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

FindOptiX.cmake
=================

OptiX provides a FindOptiX.cmake which does not follow CMake conventions, 
so what to do ?

1. copy into opticks and modify (OPTED FOR THIS)

   * NB the OptiX_INSTALL_DIR is still needed to to find the libs, 
     just the FindOptiX.cmake comes from cmake/Modules/FindOptiX.cmake 
     rather than the OptiX SDK  

2. make a fixer FindOptiX that uses the original and adds the missing pieces

   * dont like to diddle with CMAKE_MODULE_PATH so cannot do this

   ::

       CMAKE_MODULE_PATH=$(opticks-prefix)/cmake/Modules;/Developer/OptiX/SDK/CMake 

3. make a fresh FindOptiX 

::

     vimdiff /Developer/OptiX/SDK/CMake/FindOptiX.cmake $(opticks-home)/cmake/Modules/FindOptiX.cmake

EOT
}
  
om-cmake $sdir

make
make install   


$name


