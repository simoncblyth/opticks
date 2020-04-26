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

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

om-cmake $sdir 
make
make install   

$name

cat << EON > /dev/null

CMake Warning (dev) at /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake:590 (option):
  Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
  --help-policy CMP0077" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.

  For compatibility with older versions of CMake, option is clearing the
  normal variable 'CUDA_PROPAGATE_HOST_FLAGS'.
Call Stack (most recent call first):
  /Users/blyth/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/cudarap/cudarap-config.cmake:15 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/thrustrap/thrustrap-config.cmake:15 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/optixrap/optixrap-config.cmake:13 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/okop/okop-config.cmake:7 (find_dependency)
  CMakeLists.txt:6 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.

CMake Warning (dev) at /opt/local/share/cmake-3.17/Modules/FindCUDA.cmake:596 (option):
  Policy CMP0077 is not set: option() honors normal variables.  Run "cmake
  --help-policy CMP0077" for policy details.  Use the cmake_policy command to
  set the policy and suppress this warning.

  For compatibility with older versions of CMake, option is clearing the
  normal variable 'CUDA_VERBOSE_BUILD'.
Call Stack (most recent call first):
  /Users/blyth/opticks/cmake/Modules/FindOpticksCUDA.cmake:29 (find_package)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/cudarap/cudarap-config.cmake:15 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/thrustrap/thrustrap-config.cmake:15 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/optixrap/optixrap-config.cmake:13 (find_dependency)
  /opt/local/share/cmake-3.17/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /usr/local/opticks/lib/cmake/okop/okop-config.cmake:7 (find_dependency)
  CMakeLists.txt:6 (find_package)
This warning is for project developers.  Use -Wno-dev to suppress it.


EON
