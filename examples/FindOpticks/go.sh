#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

#cmake -DOpticks_DIR=$(opticks-prefix)/config -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules $sdir 
#
#  incorporating Opticks modules feels kinda like making the example part of Opticks 
#  whereas the idea is to treat these examples as unaffiliated "users" of Opticks 

cmake -DOpticks_DIR=$(opticks-prefix)/config $sdir 
make
make install   # installs executable to  /usr/local/lib/

