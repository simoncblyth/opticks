#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/examples/$name/build 

#rm -rf $bdir 
if [ ! -d "$bdir" ]; then 
    mkdir -p $bdir && cd $bdir 
    cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
                -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
                -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
                -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 
else
   cd $bdir 
fi 

pwd
make
rc=$?
[ "$rc" != "0" ] && exit $rc 

make install   


g4-
g4-export


$name


