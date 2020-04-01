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
oc-

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/opticks/$name/build 

rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


gcc -c $sdir/UseNPY.cc $(oc-cflags NPY)
gcc UseNPY.o $(oc-libs NPY) -o UseNPY 

case $(uname) in 
  Darwin) runline="DYLD_LIBRARY_PATH=$(oc-libdir) $bdir/UseNPY" ;;
   Linux) runline="LD_LIBRARY_PATH=$(oc-libdir) $bdir/UseNPY" ;;
esac

echo "runline $runline"
eval $runline

python -c "import numpy as np ; print np.load(\"$TMP/UseNPY.npy\") " 


