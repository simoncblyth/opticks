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


notes(){ cat << EON

This succeeds despite there being no boost.pc. How ? 

* no direct usage of boost headers in UseBoostRap  
* BOpticksResource.cc uses boost_filesystem but not fs in header


Check the otool -L to see which boost gets used, 
before rebuilding BoostRap its still the macports one
from /opt/local/lib/

EON
}



sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 
rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

idpath=/usr/local/opticks/opticksdata/export/DayaBay_VGDX_20140414-1300/g4_00.96ff965744a2f6b78c24e33c80d3a4cd.dae
pkg=BoostRap

echo gcc -c $sdir/Use$pkg.cc $(oc-cflags $pkg)
     gcc -c $sdir/Use$pkg.cc $(oc-cflags $pkg)
echo gcc  Use$pkg.o -o Use$pkg $(oc-libs $pkg)
     gcc  Use$pkg.o -o Use$pkg $(oc-libs $pkg)

if [ "$(uname)" == "Darwin" ]; then 
    otool -L ./Use$pkg 
    otool -L $(opticks-prefix)/lib/lib$pkg.dylib
fi

echo IDPATH=$idpath LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$pkg
     IDPATH=$idpath LD_LIBRARY_PATH=$(oc-libpath $pkg) ./Use$pkg


