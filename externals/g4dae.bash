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

g4dae-source(){   echo $BASH_SOURCE ; }
g4dae-vi(){       vi $(g4dae-source) ; }
g4dae-usage(){ cat << EOU

G4DAE : G4 COLLADA/DAE Geometry Export as an Opticks external 
=====================================================================

* see also ~/g4dae/g4d-

G4DAE Opticks Fork : As changes that make sense for an Opticks 
external under my control will break standalone usage 
have forked g4dae to g4dae-opticks.


EOU
}

g4dae-env(){     
    olocal- 
    g4- 
}

g4dae-prefix(){  echo $(opticks-prefix)/externals ; }
g4dae-base(){    echo $(g4dae-prefix)/g4dae ;  }

g4dae-dir(){     echo $(g4dae-base)/$(g4dae-name) ; }
g4dae-sdir(){    echo $(g4dae-base)/$(g4dae-name) ; }
g4dae-bdir(){    echo $(g4dae-base)/$(g4dae-name).build ; }

g4dae-c(){    cd $(g4dae-dir); }
g4dae-cd(){   cd $(g4dae-dir); }
g4dae-scd(){  cd $(g4dae-sdir); }
g4dae-bcd(){  cd $(g4dae-bdir); }

g4dae-info(){ cat << EOI

   g4dae-name    : $(g4dae-name)
   g4dae-source  : $(g4dae-source)
   g4dae-prefix  : $(g4dae-prefix)
   g4dae-base    : $(g4dae-base)
   g4dae-dir     : $(g4dae-dir)
   g4dae-bdir    : $(g4dae-bdir)

   g4dae-url     : $(g4dae-url)

   g4-cmake-dir  : $(g4-cmake-dir)   NO LONGER USED


EOI
}

g4dae-wipe(){
   local bdir=$(g4dae-bdir)
   rm -rf $bdir
}

g4dae-name(){ echo g4dae-opticks ; }
g4dae-url(){
   case $USER in
     blyth) echo ssh://hg@bitbucket.org/simoncblyth/$(g4dae-name) ;;
         *) echo https://bitbucket.org/simoncblyth/$(g4dae-name) ;;
   esac
} 

g4dae-get(){
   local iwd=$PWD
   local dir=$(dirname $(g4dae-dir)) &&  mkdir -p $dir && cd $dir
   if [ ! -d "$(g4dae-name)" ]; then 
       hg clone $(g4dae-url)
   fi 
   cd $iwd
}

g4dae-cmake(){
   local iwd=$PWD
   local bdir=$(g4dae-bdir)
   mkdir -p $bdir
   g4dae-bcd 
   cmake \
       -G "$(opticks-cmake-generator)" \
       -DCMAKE_INSTALL_PREFIX=$(g4dae-prefix) \
       -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
       -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
       -DCMAKE_BUILD_TYPE=Debug \
       $(g4dae-sdir)

   cd $iwd
}

g4dae-cmake-notes(){ cat << EON

* when building as an Opticks external seems more
  appropriate to use the higher level Opticks cmake/Modules/FindG4.cmake 
  rather than::

       -DGeant4_DIR=$(g4-cmake-dir) 
    
  this removes the possibility of different Geant4 versions being attempted
  to be used at the same time

* also means can export targets with deps via BCM  


EON
}


g4dae-make(){
   local iwd=$PWD
   g4dae-bcd 
   make $*
   cd $iwd
}

g4dae-install(){
   g4dae-make install
}

g4dae--()
{
    g4dae-get
    g4dae-cmake
    g4dae-make
    g4dae-install
}

g4dae-cls(){  
   local iwd=$PWD
   g4dae-cd 
   g4-
   g4-cls- . ${1:-G4DAEParser} ; 
   cd $iwd
}


