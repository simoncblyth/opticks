#!/bin/bash -l


sdir=$(pwd)
name=$(basename $sdir) 
bdir=/tmp/$USER/$name/build 
idir=/tmp/$USER/$name/install

rm   -rf $bdir
mkdir -p $bdir 
cd $bdir 
pwd 

cmake $sdir -DOPTICKS_CONFIG_DIR=/usr/local/opticks/config -DCMAKE_INSTALL_PREFIX=$idir

make
make install   


exe=$idir/lib/$name


echo "running exe $exe"
eval $exe

python -c "import numpy as np ; print np.load(\"$TMP/UseNPY.npy\") " 


