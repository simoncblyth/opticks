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


notes(){ cat << EON

go-cmake-with-source-tree 
   for developers who are building examples against an Opticks from source install
   (without bothering to go via a release) 
     
go-cmake-from-install-tree
   prototype-ing approach for users who are building examples against
   an Opticks binary release 
 
The distinction between these two modes comes primarily from the pfx.
Check the libs in use with ldd on the executable.
There is a further difference in the need for -DOPTICKS_PREFIX 
for the source build. This is necessary to set the runtime path for the 
executables to find their libs. 

Check the libs in use with : ldd exe

Check the runpath with : chrpath exe

EON
}


opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/examples/$name/build 
idir=/tmp/$USER/opticks/examples/$name

msg="=== $BASH_SOURCE :"

go-cmake-with-source-tree()
{
   local pfx=$1
   local sdir=$2
   local idir=$3

   cmake $sdir \
        -DCMAKE_FIND_DEBUG_MODE=0 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DOPTICKS_PREFIX=$pfx \
        -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
        -DCMAKE_INSTALL_PREFIX=$idir \
        $*

   # -DCMAKE_PREFIX_PATH="$pfx/externals;$pfx" \
      
}

go-cmake-from-install-tree()
{
   local pfx=$1
   local sdir=$2
   local idir=$3

   cmake $sdir \
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_PREFIX_PATH="$pfx/externals;$pfx" \
        -DCMAKE_MODULE_PATH=$pfx/cmake/Modules \
        -DCMAKE_INSTALL_PREFIX=$idir \
        $* 

}

go-cmake-notes(){ cat << EON

To explicitly select the Geant4 if you have multiple, try::

   -DGeant4_DIR=$(opticks-prefix)_externals/g4/lib/Geant4-10.4.2

   #go-cmake-with-source-tree $pfx $sdir $idir 
   #go-cmake-from-install-tree $pfx $sdir $idir 

EON
}




pfx=$(opticks-;opticks-prefix)        

info(){ cat << EOI

   pfx  : $pfx
   idir : $idir
   bdir : $bdir
   exe  : $exe

EOI
}

info


rm -rf $bdir 
if [ ! -d "$bdir" ]; then 
   mkdir -p $bdir && cd $bdir 
   om-
   om-cmake $sdir
else
   cd $bdir 
fi 

pwd
make
rc=$?
[ "$rc" != "0" ] && exit $rc 

make install   

setup=$pfx/bin/opticks-setup.sh 
[ ! -f $setup ] && echo "$msg MISSING setup script $setup : create with bash function opticks-setup-generate " && return 1   
source $setup

exe=$(which CerenkovMinimal)
echo $exe
ls -l $exe

if [ -n "$DEBUG" ]; then 
    case $(uname) in 
        Linux)  gdb $exe ;;
        Darwin) lldb $exe  ;;
    esac
else
    $exe
fi 

if [ "$(uname)" == "Linux" ]; then
    echo ldd $exe
    ldd $exe

    echo chrpath $exe
    chrpath $exe

elif [ "$(uname)" == "Darwin" ]; then
    otool -L $exe
fi 

info


