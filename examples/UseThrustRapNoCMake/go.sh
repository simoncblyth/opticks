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
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
 


pkg=ThrustRap

oc-dump $pkg
echo nvcc -c $sdir/Use$pkg.cu $(oc-cflags $pkg)
     nvcc -c $sdir/Use$pkg.cu $(oc-cflags $pkg)

echo gcc Use$pkg.o -o Use$pkg $(oc-libs $pkg) 
     gcc Use$pkg.o -o Use$pkg $(oc-libs $pkg) 

echo LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$pkg
     LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$pkg



cat << EOS > /dev/null

gcc UseThrustRap.o -o UseThrustRap \
       -L/usr/local/opticks/lib \
       -L/usr/local/opticks/externals/lib \
       -L/usr/local/cuda/lib \
       -lThrustRap \
        /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
     -lOpticksCore \
     -lNPY \
     -lBoostRap \
     -lOpenMeshTools \
     -lOpenMeshCore \
     -lstdc++ \
     -lDualContouringSample \
     -lstdc++ \
     -lCUDARap \
     /Developer/NVIDIA/CUDA-9.1/lib/libcudart_static.a \
     -lSysRap \
     -lstdc++ \
     -lcudart \
     -lcurand \
    -Wl,-rpath,/usr/local/cuda/lib 

EOS


nvcc-notes(){ cat << EON

nvcc doesnt like::
 
    -Wl,-rpath,/usr/local/cuda/lib 

But handles::

    --linker-options=-rpath,/usr/local/cuda/lib 
     -Xlinker=-rpath,/usr/local/cuda/lib 

So either switch the flags OR do the linking with gcc ??

EON
}




