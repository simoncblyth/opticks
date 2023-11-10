#!/bin/bash -l
sdir=$(pwd)
name=UseRelease

TMP=${TMP:-/tmp/$USER/opticks}
bdir=$TMP/${name}.build 
idir=$TMP/${name}.install
bin=$idir/lib/$name

vars="BASH_SOURCE TMP sdir bdir idir bin OPTICKS_PREFIX"
for var in $vars ; do printf "%25s : %s\n" "$var" "${!var}" ; done 

echo === $BASH_SOURCE : CMAKE_PREFIX_PATH
echo $CMAKE_PREFIX_PATH | tr ":" "\n"

rm -rf $bdir 
rm -rf $idir 

# THIS SHOULD BE SET VIA THE ENVVAR     
# -DCMAKE_PREFIX_PATH="$OPTICKS_PREFIX/externals;$OPTICKS_PREFIX" \

if [ ! -d "$bdir" ]; then 
   mkdir -p $bdir && cd $bdir 
   cmake $sdir \
      -DCMAKE_BUILD_TYPE=Debug \
      -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
      -DCMAKE_MODULE_PATH=$OPTICKS_PREFIX/cmake/Modules \
      -DCMAKE_INSTALL_PREFIX=$idir
else
   cd $bdir 
fi 

pwd
make
rc=$?
[ "$rc" != "0" ] && exit $rc 

make install   

echo === $BASH_SOURCE : $bin
$bin

