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

rm -rf $bdir && mkdir -p $bdir 
cd $bdir && pwd 
#ls -l 

if [ ! -f CMakeCache.txt ]; then  
    om-cmake $sdir 
fi

make
make install   

$name

ls -l /tmp/$name.ppm

open /tmp/$name.ppm



cat << EON > /dev/null

Compilation fails with OptiX 501

/Users/blyth/opticks/examples/UseOptiXGeometryTriangles/UseOptiXGeometryTriangles.cu(112): error: identifier "rtGetPrimitiveIndex" is undefined

The CMake find_package with a version 6.0.0 fails to notify that got too old an OptiX 


EON


