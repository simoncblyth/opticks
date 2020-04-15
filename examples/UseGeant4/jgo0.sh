#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

 
jexlib=$HOME/junotop/ExternalLibs

jg4home=$jexlib/Geant4/10.05.p01
jg4lib=$jg4home/lib64
jg4cmakedir=$jg4home/lib64/Geant4-10.5.1

jxchome=$jexlib/Xercesc/3.2.2
jxclib=$jxchome/lib




cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
            -DGeant4_DIR=$jg4cmakedir    

make
make install   

LD_LIBRARY_PATH=$jg4lib:$jxclib UseGeant4


