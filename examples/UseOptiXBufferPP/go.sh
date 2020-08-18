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

  
om-cmake $sdir 
#  -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) 

make
[ $? -ne 0 ] && echo make ERROR && exit 1 

make install   

$name --help
#$name bufferTest.cu bufferTest_readOnly
$name bufferTest.cu bufferTest_readWrite


ptx=$(opticks-prefix)/installcache/PTX/${name}_generated_bufferTest.cu.ptx
ls -l $ptx
ptx.py $ptx --all | c++filt

