#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

* OptiX_INSTALL_DIR cmake argument is still needed to to find the libs
  for any package downstream from OptiX  ... it would be better
  to encapsulate this into the persisted target ?

EOT
}
  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
            -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) 

make
make install   

