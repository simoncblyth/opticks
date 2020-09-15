#!/bin/bash 

notes(){ cat << EON

To test without inheriting the invoking environment::

   env -i PATH="$PATH" ./go.sh 
   env -i PATH="$PATH" CMAKE_PREFIX_PATH=/usr/local/opticks_externals/g4      ./go.sh 
   env -i PATH="$PATH" CMAKE_PREFIX_PATH=/usr/local/opticks_externals/g4_1042:/usr/local/opticks_externals ./go.sh 
   env -i PATH="$PATH" CMAKE_PREFIX_PATH=/usr/local/opticks_externals/g4_1062:/usr/local/opticks_externals ./go.sh 


BCM and Geant4 installs are required, get those with eg::

    bcm-
    BCM_PREFIX=/usr/local/opticks_externals bcm--

    g4-
    g4--1042 
    g4--1062


Notice the CMAKE_MODULE_PATH to pick up the FindG4.cmake for this source dir.


EON
}


sdir=$(pwd)
name=$(basename $sdir)
idir=/tmp/$USER/opticks/$name
bdir=/tmp/$USER/opticks/$name.build 
echo sdir $sdir name $name idir $idir bdir $bdir

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 
rm -rf $idir

env
echo CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

cmake $sdir \
    -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$idir \
    -DCMAKE_MODULE_PATH=$sdir

make
make install   


