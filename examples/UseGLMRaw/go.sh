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


usage(){ cat << EOU

Simple script to exercise CMake machinery, 
running this script does:

* cmake configuration, generating Makefiles into bdir beneath /tmp
* building 
* installing into opticks-prefix, so ensure the basename of the directory is unique
* run the resulting executable

Everything is done on every invokation.


EOU
}


opticks-

home=$(opticks-home)
prefix=$(opticks-prefix)

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 
idir=/tmp/$USER/$name/install

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$prefix -DCMAKE_MODULE_PATH=$home/cmake/Modules

make
make install   

exe=$prefix/lib/Test$name


if [ -f "$exe" ]; then
    ls -l $exe
    echo running installed exe $exe
    $exe
else 
    echo failed to install exe to $exe 
fi 



