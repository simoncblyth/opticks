#!/bin/bash -l

opticks-
opticks-boost-info

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
     -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

cat << EOC > /dev/null

     -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
     -DBOOST_LIBRARYDIR=$(opticks-boost-libdir) \
     -DBoost_USE_STATIC_LIBS=1 \
     -DBoost_USE_DEBUG_RUNTIME=0 \
     -DBoost_NO_SYSTEM_PATHS=1 \
     -DBoost_DEBUG=0

EOC

make
make install   


if [ "$(uname)" == "Linux" ]; then
   ldd $(opticks-prefix)/lib64/libUseBoost.so
fi 

