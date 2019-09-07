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
opticks-id 
prefix=$(opticks-prefix-tmp)

msg="=== $0 :"
cat << EOM

$msg integrated build : for now into opticks-prefix-tmp : $(opticks-prefix-tmp)

EOM

if [ ! -d "$prefix" ]; then 
   mkdir -p $prefix
fi 
cd $prefix

## hmm tis awkward having the externals and opticksdata inside the prefix
## when dealing with multiple Opticks versions 

if [ -d externals ]; then 
   echo found externals dir or link inside the prefix
else
   echo creating link to externals
   ln -s ../opticks/externals
fi

if [ -d opticksdata ]; then 
   echo found opticksdata dir inside the prefix
else
   echo creating link to opticksdata
   ln -s ../opticks/opticksdata
fi

pwd  ; ls -l 


cd $(opticks-home)

sdir=$(pwd)
bdir=$prefix/build

if [ "$1" == "clean" ]; then
   echo $0 $1 : remove bdir $bdir
   rm -rf $bdir 
fi 
mkdir -p $bdir && cd $bdir && pwd 

echo $msg sdir : $sdir 
echo $msg bdir : $bdir 
echo $msg pwd : $(pwd) 


t0=$(date +"%s")
cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$prefix \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
     \
    -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) \
    -DCOMPUTE_CAPABILITY=$(opticks-compute-capability)

rc=$?
if [ "$rc" != "0" ]; then
    echo $msg cmake rc $rc non-zero 
    exit $rc
fi
t1=$(date +"%s")
d1=$(( t1 - t0 ))

echo $msg cmake configuring $d1 seconds ie $(( d1/60 )) minutes 

#exit 1


make

t2=$(date +"%s")
d2=$(( t2 - t1 ))
echo $msg make building $d2 seconds ie $(( d2/60 )) minutes 

if [ "$(uname)" == "Darwin" ]; then
   echo "kludge sleeping for 2s"
   sleep 2
fi 

make install   
t3=$(date +"%s")
d3=$(( t3 - t2 ))
echo $msg make installing $d3 seconds ie $(( d3/60 )) minutes 



printf "%s %30s %s \n" "$msg" "cmake configuring" "$d1 seconds ie $(( d1/60 )) minutes" 
printf "%s %30s %s \n" "$msg" "make building"     "$d2 seconds ie $(( d2/60 )) minutes" 
printf "%s %30s %s \n" "$msg" "make installing"   "$d3 seconds ie $(( d3/60 )) minutes" 




