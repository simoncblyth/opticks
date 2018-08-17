#!/bin/bash -l

opticks-

sdir=$(pwd)
bdir=/tmp/$USER/opticks/$(basename $sdir)/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


cmake_resolution_note(){ cat << EON

One of the prefixes provided in either install_prefix or prefix_path 
needs to be $(opticks-prefix) in order for CMake resolution 
to find the okconf-config.cmake

For example the below setting would fail to find okconf-config.cmake::

    install_prefix=/tmp             

Unless also have::

    prefix_path="$(opticks-prefix)/externals;$(opticks-prefix)"

The standard used everywhere in Opticks is::

    install_prefix=$(opticks-prefix)
    prefix_path="$(opticks-prefix)/externals

EON
}

install_prefix=$(opticks-prefix)  
prefix_path=$(opticks-prefix)/externals


cmake $sdir \
     -DCMAKE_BUILD_TYPE=$(opticks-buildtype) \
     -DCMAKE_INSTALL_PREFIX=${install_prefix} \
     -DCMAKE_PREFIX_PATH=${prefix_path} \
     -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

make
make install   

