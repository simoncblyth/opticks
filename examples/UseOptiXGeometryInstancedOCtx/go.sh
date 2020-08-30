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


remake=0

if [ $remake -eq 1 -o ! -d $bdir ]; then  
    rm -rf $bdir && mkdir -p $bdir 
    cd $bdir && pwd 
    om-cmake $sdir
else
    cd $bdir && pwd 
fi 


make
[ ! $? -eq 0 ] && echo build error && exit 1

make install   

earth=$HOME/opticks_refs/Earth_Albedo_8192_4096.ppm
if [ -f "$earth" ]; then 
    path=$earth  
else
    path=/tmp/SPPMTest.ppm
fi
path=/tmp/SPPMTest.ppm


echo $name $path
NPYBase=info $name $path
[ ! $? -eq 0 ] && echo runtime error && exit 1



outpath=/tmp/$USER/opticks/$name/out.ppm
ls -l $outpath
date

if [ -n "$SSH_TTY" ]; then 
    echo remote running : outpath $outpath
else 
    echo local running : open outpath $outpath
    open $outpath
fi





