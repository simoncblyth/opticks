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

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


pkg=OKG4

echo gcc -c $sdir/Use$pkg.cc $(oc --cflags $pkg)
     gcc -c $sdir/Use$pkg.cc $(oc --cflags $pkg)

#echo gcc Use$pkg.o -o Use$pkg $(oc --libs $pkg) 
#     gcc Use$pkg.o -o Use$pkg $(oc --libs $pkg) 


check()
{
gcc UseOKG4.o -o UseOKG4 \
      /usr/local/cuda/lib64/libcudart_static.a \
         /usr/lib64/librt.so \
        -L/home/blyth/local/opticks/lib64 \
        -L/home/blyth/local/opticks/externals/lib \
        -L/home/blyth/local/opticks/externals/lib64 \
        -L/home/blyth/local/opticks/externals/OptiX_650/lib64 \
        -L/usr/local/cuda/lib64 \
        -L/home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bin/../lib64 \
        -L/home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/lib \
        -lOKG4 -lOK -lCFG4 -lOpticksGL -lOGLRap -lOKOP \
         -lImGui -lGLEW -lglfw \
           -lGLU -lGL \
        -lOptiXRap -loptix -loptixu -loptix_prime \
           -lExtG4 -lOpticksGeo -lThrustRap \
           -lG4OpenGL -lG4gl2ps -lG4Tree -lG4FR -lG4GMocren \
            -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling \
              -lG4interfaces -lG4persistency -lG4analysis -lG4error_propagation -lG4readout \
           -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits \
            -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms \
                -lG4global -lG4clhep -lG4zlib \
              -lAssimpRap -lOpenMeshRap -lassimp \
                -lGGeo -lYoctoGLRap -lOpticksCore -lCUDARap \
                   -lNPY -lBoostRap \
                 -lOpenMeshTools -lOpenMeshCore \
                    -lYoctoGL -lImplicitMesher \
                   -lDualContouringSample \
                -lSysRap \
               -lxerces-c \
                -lssl -lcrypto \
                        -lm \
                  -lpthread -ldl \
                 -lcudart -lcurand \
                   -lstdc++ \
                   -lOKConf

}

#check



#echo ./Use$pkg
#     ./Use$pkg

#gdb ./Use$pkg

check2()
{
gcc UseOKG4.o -o UseOKG4 \
      /usr/local/cuda/lib64/libcudart_static.a \
         /usr/lib64/librt.so \
        -L/home/blyth/local/opticks/lib64 \
        -L/home/blyth/local/opticks/externals/lib \
        -L/home/blyth/local/opticks/externals/lib64 \
        -L/home/blyth/local/opticks/externals/OptiX_650/lib64 \
        -L/usr/local/cuda/lib64 \
        -L/home/blyth/junotop/ExternalLibs/Geant4/10.05.p01/bin/../lib64 \
        -L/home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/lib \
-lOKG4 -lOK -lOpticksGL -lOGLRap -lGLEW -lImGui -lGLEW -lglfw -lOKOP -lOptiXRap -loptix -loptixu -loptix_prime -lstdc++ -lCFG4 -lExtG4 -lG4Tree -lG4FR -lG4GMocren -lG4visHepRep -lG4RayTracer -lG4VRML -lG4vis_management -lG4modeling -lG4interfaces -lG4persistency -lG4analysis -lG4error_propagation -lG4readout -lG4physicslists -lG4run -lG4event -lG4tracking -lG4parmodels -lG4processes -lG4digits_hits -lG4track -lG4particles -lG4geometry -lG4materials -lG4graphics_reps -lG4intercoms -lG4global -lG4clhep -lG4zlib -lstdc++ -lxerces-c -lOpticksGeo -lAssimpRap -lassimp -lOpenMeshRap -lGGeo -lYoctoGLRap -lThrustRap /usr/local/cuda/lib64/libcudart_static.a -Wl,-rpath,/usr/local/cuda/lib64 -lOpticksCore -lNPY -lBoostRap -lOpenMeshTools -lOpenMeshCore -lstdc++ -lm -lYoctoGL -lstdc++ -lImplicitMesher -lDualContouringSample -lstdc++ -lCUDARap /usr/local/cuda/lib64/libcudart_static.a -Wl,-rpath,/usr/local/cuda/lib64 -lSysRap -lstdc++ -lcudart -lcurand -lOKConf


 
}
check2

