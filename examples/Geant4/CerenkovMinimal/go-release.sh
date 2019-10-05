#!/bin/bash -l


pfx=$(opticks-release-prefix)
rc=$?
[ $rc -ne 0 ] && echo $0 requires opticks-release-prefix from user_setup && exit $rc


sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/examples/$name/build 
idir=$(opticks-release-user-home)   # OPTICKS_USER_HOME falling back to HOME
# /tmp is not same on GPU cluster gateway and nodes, so cannot install there 

echo pfx $pfx

rm -rf $bdir 
if [ ! -d "$bdir" ]; then 
   mkdir -p $bdir && cd $bdir 
   cmake3 $sdir \
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


cat << EOX

# how to run example

unset OPTICKS_GEOCACHE_PREFIX 

# for simple geometry testing must unset the geocache prefix
# which will cause the default to be used ~/.opticks
# otherwise will see permission errors on attempting 
# to write into the shared cache

$exe

EOX





