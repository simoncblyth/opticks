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

thoughts(){ cat << EOT

Is this userland or developer land ?  

In userland should be using the setup script directly 
and not relying on bash functions.

EOT
}


NAME=$(basename $BASH_SOURCE)
MSG="=== $NAME :"

echo $MSG opticks-
opticks-

echo $MSG om- invokes om-env which invokes oe- running the setup
om-
rc=$?

if [ ! $rc -eq 0 ]; then 
   echo $MSG om- setup failed rc $rc
   exit $rc
fi 


echo $MSG oe-info
oe-info


sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir 
mkdir -p $bdir && cd $bdir && pwd 


echo $MSG om-cmake
om-cmake $sdir


#cmake $sdir \
#     -G "$(om-cmake-generator)" \
#     -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
#     -DOPTICKS_PREFIX=$(om-prefix) \
#     -DCMAKE_INSTALL_PREFIX=$(om-prefix) \
#     -DCMAKE_MODULE_PATH=$(om-home)/cmake/Modules



echo $MSG make
make
[ "$(uname)" == "Darwin" ] && echo "Kludge sleep 2s" && sleep 2 

echo $MSG make install
make install   

bin=$(which $name)
ls -l $bin

echo $MSG $bin
$bin


