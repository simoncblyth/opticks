#!/bin/bash -l

usage(){ cat << EOU

Fail to find GLM and probably others without::

   -DOPTICKS_PREFIX=$(opticks-prefix) 


EOU
}


opticks-

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 
rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


cmake_()
{
    echo $FUNCNAME
    cmake $sdir \
        -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
        -DCMAKE_BUILD_TYPE=Debug \
                                 \
        -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
        -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
        -DOPTICKS_PREFIX=$(opticks-prefix) 
}

cmake_no_module()
{
    : skipping the module path results in a clean fail
    echo $FUNCNAME
    cmake $sdir \
        -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
        -DCMAKE_BUILD_TYPE=Debug \
                                 \
        -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
        -DOPTICKS_PREFIX=$(opticks-prefix) 
}

cmake_no_opticks_prefix()
{
    : skipping the -DOPTICKS_PREFIX results in failed finding GLM
    echo $FUNCNAME
    cmake $sdir \
        -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
        -DCMAKE_BUILD_TYPE=Debug \
                                 \
        -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
        -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 
}


cmake_no_prefix_path()
{
    : succeeds as the full prefix path is coming in via envvar anyhow 
    echo $FUNCNAME
    cmake $sdir \
        -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
        -DCMAKE_BUILD_TYPE=Debug \
                                 \
        -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules  \
        -DOPTICKS_PREFIX=$(opticks-prefix) 
}


om_cmake()
{
    echo $FUNCNAME
    om-
    om-cmake $sdir
}


#cmake_
#cmake_no_module
#cmake_no_opticks_prefix
cmake_no_prefix_path
#om_cmake





