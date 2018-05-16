#!/bin/bash -l


sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake -DOPTICKS_CONFIG_DIR=/usr/local/opticks/config $sdir 

make
make install   # installs executable to  /usr/local/lib/

