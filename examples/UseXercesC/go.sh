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

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


#unset CMAKE_PREFIX_PATH
#unset PKG_CONFIG_PATH

echo CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

  
cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     -DOPTICKS_PREFIX=$(opticks-prefix)



echo PKG_CONFIG_PATH
echo $PKG_CONFIG_PATH | tr ":" "\n"

echo pkg-config --libs xerces-c
     pkg-config --libs xerces-c

echo pkg-config --cflags xerces-c
     pkg-config --cflags xerces-c



#make
#make install   

cat << EON > /dev/null

Without CMAKE_PREFIX_PATH, PKG_CONFIG_PATH this finds::

    -- Found XercesC: /usr/lib64/libxerces-c.so (found version "3.1.1") 
    -- XercesC_FOUND        : TRUE 
    -- XercesC_VERSION      : 3.1.1 
    -- XercesC_INCLUDE_DIRS : /usr/include 
    -- XercesC_LIBRARIES    : /usr/lib64/libxerces-c.so 

    pkg-config --libs xerces-c
    -lxerces-c  
    pkg-config --cflags xerces-c
     

with the $JUNOTOP/bashrc.sh CMAKE_PREFIX_PATH, PKG_CONFIG_PATH gives::

    -- Found XercesC: /home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/lib/libxerces-c.so (found version "3.2.2") 
    -- XercesC_FOUND        : TRUE 
    -- XercesC_VERSION      : 3.2.2 
    -- XercesC_INCLUDE_DIRS : /home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include 
    -- XercesC_LIBRARIES    : /home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/lib/libxerces-c.so 

    pkg-config --libs xerces-c
    -L/home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/lib -lxerces-c  
    pkg-config --cflags xerces-c
    -I/home/blyth/junotop/ExternalLibs/Xercesc/3.2.2/include  


Note that the -DCMAKE_PREFIX_PATH argument does not occlude the envvar 


EON



