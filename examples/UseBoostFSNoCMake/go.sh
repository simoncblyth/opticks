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


pkg=Boost
name=${pkg}FS

oc -n $pkg


echo gcc -c $sdir/Use$name.cc $(oc -cflags $pkg)
     gcc -c $sdir/Use$name.cc $(oc -cflags $pkg)
echo gcc Use$name.o -o Use$name $(oc -libs $pkg) 
     gcc Use$name.o -o Use$name $(oc -libs $pkg) 
echo ./Use$name
     ./Use$name

# this fails with system boost for lack of boost.pc
# as the pkg_config.py resolution is not looking for /usr/lib64/pkgconfig/boost.pc




