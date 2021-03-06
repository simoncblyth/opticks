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

make
[ ! $? -eq 0 ] && echo build error && exit 1

make install   

earth=$HOME/opticks_refs/Earth_Albedo_8192_4096.ppm
gradient=/tmp/SPPMTest_MakeTestImage.ppm 
if [ -f "$earth" ]; then 
    path=$earth  
else
    path=$gradient
fi
path=$gradient


echo $name $path
$name $path
[ ! $? -eq 0 ] && echo runtime error && exit 1


outpath=/tmp/$USER/opticks/$name/out.ppm

if [ -n "$SSH_TTY" ]; then 
    echo remote running : outpath $outpath
else 
    echo local running : open outpath $outpath
    open $outpath
fi






