#!/bin/bash -l

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/examples/$name/build 
idir=/tmp/$USER/opticks/examples/$name

pfx=$(opticks-release-prefix)
echo pfx $pfx

rm -rf $bdir 
if [ ! -d "$bdir" ]; then 
   mkdir -p $bdir && cd $bdir 
   cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_PREFIX_PATH="$pfx/externals;$pfx" \
     -DCMAKE_MODULE_PATH=$pfx/cmake/Modules \
     -DCMAKE_INSTALL_PREFIX=$idir \
     -DGeant4_DIR=$(opticks-envg4-Geant4_DIR)
else
   cd $bdir 
fi 

pwd
make
rc=$?
[ "$rc" != "0" ] && exit $rc 

make install   


exe=$idir/lib/$name
echo $exe
$exe



