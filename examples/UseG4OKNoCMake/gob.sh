#!/bin/bash
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
Minimal environment test::

   env -i HOME=$HOME OPTICKS_PREFIX=$OPTICKS_PREFIX OPTICKS_OPTIX_PREFIX=/usr/local/optix PATH=/usr/local/cuda/bin:/opt/local/bin:/usr/bin:/bin ./gob.sh

Above commandline allows to check opticks setup from a minimal environment. 
Note that /usr/local/optix and /usr/local/cuda are symbolic links the optix one being non-standard.

EON
}

env 
source $OPTICKS_PREFIX/bin/opticks-setup.sh 


sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 

pkg=G4OK

echo gcc -c $sdir/Use$pkg.cc $(oc --cflags $pkg)
     gcc -c $sdir/Use$pkg.cc $(oc --cflags $pkg)
echo gcc Use$pkg.o -o Use$pkg $(oc --libs $pkg) 
     gcc Use$pkg.o -o Use$pkg $(oc --libs $pkg) 
echo ./Use$pkg
     ./Use$pkg





