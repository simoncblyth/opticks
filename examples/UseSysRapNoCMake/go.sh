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


# manual 
#gcc -c $sdir/UseSysRap.cc -I$(opticks-prefix)/include/SysRap 
#gcc  UseSysRap.o $sdir/TestUseSysRap.cc -o TestUseSysRap -L$(opticks-prefix)/lib -lSysRap



# bash
#gcc -c $sdir/UseSysRap.cc $(oc-cflags SysRap)
#gcc  UseSysRap.o $sdir/TestUseSysRap.cc -o TestUseSysRap $(oc-libs SysRap)
#DYLD_LIBRARY_PATH=$(oc-libdir) $bdir/TestUseSysRap

# py : generated strings like $(opticks-prefix) do not interpolated, so have done that in the python emitting absolutes here
echo gcc -c $sdir/UseSysRap.cc $(oc.py SysRap --flags)  
     gcc -c $sdir/UseSysRap.cc $(oc.py SysRap --flags)

echo gcc  UseSysRap.o $sdir/TestUseSysRap.cc -o TestUseSysRap $(oc.py SysRap --libs)
     gcc  UseSysRap.o $sdir/TestUseSysRap.cc -o TestUseSysRap $(oc.py SysRap --libs)

echo DYLD_LIBRARY_PATH=$(oc.py --libdir) $bdir/TestUseSysRap
     DYLD_LIBRARY_PATH=$(oc.py --libdir) $bdir/TestUseSysRap



