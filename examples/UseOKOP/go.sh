#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


thoughts(){ cat << EOT

* OptiX_INSTALL_DIR cmake argument is still needed to to find the libs
  for any package downstream from OptiX  ... it would be better
  to encapsulate this into the persisted target ?


Try to live without::

            -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) 

Gives::

    CMake Error at /Users/blyth/opticks-cmake-overhaul/cmake/Modules/FindOptiX.cmake:75 (message):
      optix library not found.  Try adding cmake argument:
      -DOptiX_INSTALL_DIR=$(opticks-optix-install-dir) (all packages downstream from OptiX need this)
    Call Stack (most recent call first):
      /Users/blyth/opticks-cmake-overhaul/cmake/Modules/FindOptiX.cmake:87 (OptiX_report_error)
      /opt/local/share/cmake-3.11/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /usr/local/opticks-cmake-overhaul/lib/cmake/optixrap/optixrap-config.cmake:4 (find_dependency)
      /opt/local/share/cmake-3.11/Modules/CMakeFindDependencyMacro.cmake:48 (find_package)
      /usr/local/opticks-cmake-overhaul/lib/cmake/okop/okop-config.cmake:4 (find_dependency)
      CMakeLists.txt:7 (find_package)


EOT
}
  
cmake $sdir -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
            -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
            -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 




make
make install   

