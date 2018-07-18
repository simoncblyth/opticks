#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

FindOptiX.cmake
=================

OptiX provides a FindOptiX.cmake which does not follow CMake conventions, 
so what to do ?

1. copy into opticks and modify (OPTED FOR THIS)

   * NB the OptiX_INSTALL_DIR is still needed to to find the libs, 
     just the FindOptiX.cmake comes from cmake/Modules/FindOptiX.cmake 
     rather than the OptiX SDK  

2. make a fixer FindOptiX that uses the original and adds the missing pieces

   * dont like to diddle with CMAKE_MODULE_PATH so cannot do this

   ::

       CMAKE_MODULE_PATH=$(opticks-prefix)/cmake/Modules;/Developer/OptiX/SDK/CMake 

3. make a fresh FindOptiX 

::

     vimdiff /Developer/OptiX/SDK/CMake/FindOptiX.cmake $(opticks-home)/cmake/Modules/FindOptiX.cmake

EOT
}
  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
            -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) 

make
make install   

