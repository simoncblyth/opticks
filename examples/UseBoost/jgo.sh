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
opticks-boost-info

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

#unset JUNOTOP
if [ -z "$JUNOTOP" ]; then 
  echo no JUNOTOP
else
  source $JUNOTOP/bashrc.sh
  if [ ! -d "$JUNO_EXTLIB_Boost_HOME" ]; then 
       echo missing JUNO_EXTLIB_Boost_HOME
       exit 1 
  fi
  env | grep JUNO_EXTLIB_Boost
fi


#unset CMAKE_PREFIX_PATH
#unset LD_LIBRARY_PATH

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:$(opticks-prefix)/externals

echo $CMAKE_PREFIX_PATH | tr ":" "\n"


cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     -DOPTICKS_PREFIX=$(opticks-prefix)

# no prefix path arg means are sensitive to the envvar, so will find the juno Boost 


cat << EOC > /dev/null

     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \

Huh : somehow CMake finds the juno boost without any assistance ?


     -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
     -DBOOST_LIBRARYDIR=$(opticks-boost-libdir) \
     -DBoost_USE_STATIC_LIBS=1 \
     -DBoost_USE_DEBUG_RUNTIME=0 \
     -DBoost_NO_SYSTEM_PATHS=1 \
     -DBoost_DEBUG=0



[blyth@localhost Modules]$ locate FindBoost.cmake
/home/blyth/junotop/ExternalLibs/Build/cmake-3.15.2/Modules/FindBoost.cmake
/home/blyth/junotop/ExternalLibs/Cmake/3.15.2/share/cmake-3.15/Modules/FindBoost.cmake
/usr/share/cmake3/Modules/FindBoost.cmake




EOC

make
make install   


if [ "$(uname)" == "Linux" ]; then
   ldd $(opticks-prefix)/lib64/libUseBoost.so
fi 

