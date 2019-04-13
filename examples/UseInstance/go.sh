#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 

make
make install   

echo executing ${name}Test

om-
#om-run UseInstanceTest
om-run OneTriangleTest 


notes(){ cat << EON

Why RPATH not working here on Darwin, it works for tests from Opticks subs ?

   DYLD_LIBRARY_PATH=/usr/local/opticks/lib OneTriangleTest

EON
}


