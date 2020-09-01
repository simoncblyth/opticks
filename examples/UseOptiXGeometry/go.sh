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
ls -l 

om-cmake $sdir 


make
[ $? -ne 0 ] && echo $0 : buildtime error && exit 1
make install   

#geocu=sphere.cu
#geocu=csg_intersect_primitive.cu
geocu=csg_intersect_part.cu

cmdline="$name $geocu"
echo $cmdline
eval $cmdline

[ $? -ne 0 ] && echo $0 : runtime error && exit 1


outpath=/tmp/$name.ppm
ls -l $outpath

if [ -n "$SSH_TTY" ]; then 
    echo remote running : outpath $outpath
else 
    echo local running : open outpath $outpath
    open $outpath
fi



