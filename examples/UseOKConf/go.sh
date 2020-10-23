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


cmake_resolution_note(){ cat << EON

One of the prefixes provided in either install_prefix or prefix_path 
needs to be $(opticks-prefix) in order for CMake resolution 
to find the okconf-config.cmake

For example the below setting would fail to find okconf-config.cmake::

    install_prefix=/tmp             

Unless also have::

    prefix_path="$(opticks-prefix)/externals;$(opticks-prefix)"

The standard used everywhere in Opticks is::

    install_prefix=$(opticks-prefix)
    prefix_path="$(opticks-prefix)/externals

EON
}

manual=1

if [ $manual -eq 1 ]; then 
   cmake $sdir \
     -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

else
   om-cmake $sdir 
fi 




make
make install   

