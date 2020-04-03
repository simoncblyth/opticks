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

idpath=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae

# bash
#gcc -c $sdir/UseBoostRap.cc $(oc-cflags BoostRap)
#gcc  UseBoostRap.o -o UseBoostRap $(oc-libs BoostRap)
#IDPATH=$idpath DYLD_LIBRARY_PATH=$(opticks-prefix)/lib $bdir/UseBoostRap

# py 
gcc -c $sdir/UseBoostRap.cc $(oc.py BoostRap --flags)
gcc  UseBoostRap.o -o UseBoostRap $(oc.py BoostRap --libs)
IDPATH=$idpath DYLD_LIBRARY_PATH=$(oc.py --libpath) $bdir/UseBoostRap


