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
snam=$(basename $sdir)
bdir=/tmp/$USER/opticks/$snam/build 
rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

pkg=${snam/NoCMake}
pkg=${pkg/Use}

echo snam $snam pkg $pkg 
ccs=$sdir/*.cc

num_main=0

: compile cc which do not have mains
for cc in $ccs
do 
    if [[ "$(grep ^int\ main $cc)" == "int main"* ]]; then 
        num_main=$(( ${num_main} + 1 ))
    else
        echo gcc -c $cc $(oc -cflags $pkg) -fpic
             gcc -c $cc $(oc -cflags $pkg) -fpic
    fi
done

sfx=""
case $(uname) in 
  Darwin) sfx=dylib ;; 
   Linux) sfx=so ;; 
esac

: create a library of the non-mains
echo gcc -shared -o libUse$pkg.$sfx $(ls *.o) $(oc -libs $pkg)
     gcc -shared -o libUse$pkg.$sfx $(ls *.o) $(oc -libs $pkg)


: compile and link the mains and run them 
for cc in $ccs
do 
    if [[ "$(grep ^int\ main $cc)" == "int main"* ]]; then 
        main=$cc
        name=$(basename $cc)
        name=${name/.cc}
        echo main $main name $name 
        echo gcc -o $name $main -L$(pwd) -lUse$pkg $(oc -libs $pkg) 
             gcc -o $name $main -L$(pwd) -lUse$pkg $(oc -libs $pkg) 
        echo ./$name
             ./$name
    fi 
done 



