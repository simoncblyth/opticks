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

optix-prefix(){ echo $(opticks-prefix)/externals/OptiX_700 ; }
[ ! -d "$(optix-prefix)" ] && echo no optix-prefix dir $(optix-prefix) && exit 0 



sdir=$(pwd)

name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1

cd $bdir && pwd 



if [ ! -f CMakeCache.txt ]; then 
 
    cmake $sdir \
          -DCMAKE_BUILD_TYPE=Debug \
          -DCMAKE_MODULE_PATH=$(optix-prefix)/SDK/CMake \
          -DOptiX_INSTALL_DIR=$(optix-prefix) \
          -DCMAKE_INSTALL_PREFIX=$(opticks-prefix)
            
fi 

make
make install   




