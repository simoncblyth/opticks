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


om-

#om-export
#om-export-info

om-export-test   # artifical setting CMAKE_PREFIX_PATH and PKG_CONFIG_PATH to test resolution

libdir=$(find_package.py boost --libdir --index 0)

oc-

pkg=Boost
name=${pkg}FS

find_package.py $pkg
pkg_config.py $pkg 


#PKG_CONFIG_PATH=$(om-pkg-config-path-reversed)
#pkg_config.py $pkg 


echo gcc -c $sdir/Use$name.cc $(oc-cflags $pkg)
     gcc -c $sdir/Use$name.cc $(oc-cflags $pkg)
echo gcc Use$name.o -o Use$name $(oc-libs $pkg) #-lpython2.7
     gcc Use$name.o -o Use$name $(oc-libs $pkg) #-lpython2.7

# with boost-python present in the libs get missing symbol without -lpython2.7
# now adding this in the boost-pcc libs list when a boost_python lib is seen


if [ "$(uname)" == "Darwin" ]; then 
    echo DYLD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$name
         DYLD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$name
else
    echo LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$name
         LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$name
fi



