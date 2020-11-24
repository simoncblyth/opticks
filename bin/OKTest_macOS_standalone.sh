#!/bin/bash

usage(){ cat << EOU
OKTest_macOS_standalone.sh 
============================

This attempts to be a standalone run of OKTest without using bash login 
setup for use with macOS dtruss system call monitoring and similar 
situations where a standalone script is needed.

That forces this to be installation specific.

NB on Linux can use strace

See notes/issues/macos-dtruss-monitor-file-opens.rst

::

    sudo dtruss -f -t open /opt/local/bin/bash /usr/local/opticks/bin/OKTest_macOS_standalone.sh 



EOU
}


opticks-prepend-prefix () 
{ 
    local msg="=== $FUNCNAME :";
    local prefix=$1;
    [ ! -d "$prefix" ] && echo $msg prefix $prefix does not exist && return 1;
    local bindir=$prefix/bin;
    local libdir="";
    if [ -d "$prefix/lib64" ]; then
        libdir=$prefix/lib64;
    else
        if [ -d "$prefix/lib" ]; then
            libdir=$prefix/lib;
        fi;
    fi;
    [ -z "$libdir" ] && echo $msg FAILED to find libdir under prefix $prefix && return 2;
    if [ -z "$CMAKE_PREFIX_PATH" ]; then
        export CMAKE_PREFIX_PATH=$prefix;
    else
        export CMAKE_PREFIX_PATH=$prefix:$CMAKE_PREFIX_PATH;
    fi;
    if [ -z "$PKG_CONFIG_PATH" ]; then
        export PKG_CONFIG_PATH=$libdir/pkgconfig;
    else
        export PKG_CONFIG_PATH=$libdir/pkgconfig:$PKG_CONFIG_PATH;
    fi;
    if [ -d "$bindir" ]; then
        if [ -z "$PATH" ]; then
            export PATH=$bindir;
        else
            export PATH=$bindir:$PATH;
        fi;
    fi;
    case $(uname) in 
        Darwin)
            libpathvar=DYLD_LIBRARY_PATH
        ;;
        Linux)
            libpathvar=LD_LIBRARY_PATH
        ;;
    esac;
    if [ -z "${!libpathvar}" ]; then
        export ${libpathvar}=$libdir;
    else
        export ${libpathvar}=$libdir:${!libpathvar};
    fi
}

export OPTICKS_PREFIX=$(dirname $(dirname $BASH_SOURCE))

export OPTICKS_CUDA_PREFIX=/usr/local/cuda
export OPTICKS_OPTIX_PREFIX=/usr/local/optix
export OPTICKS_COMPUTE_CAPABILITY=30

opticks-prepend-prefix /usr/local/opticks_externals/clhep
opticks-prepend-prefix /usr/local/opticks_externals/xercesc
opticks-prepend-prefix /usr/local/opticks_externals/g4 
opticks-prepend-prefix /usr/local/opticks_externals/boost

#source $OPTICKS_PREFIX/bin/opticks-setup.sh 1> /dev/null
source $OPTICKS_PREFIX/bin/opticks-setup.sh

export OPTICKS_GEOCACHE_PREFIX=/usr/local/opticks    ## override default opticks-geocache-prefix of ~/.opticks
export OPTICKS_KEY=OKX4Test.X4PhysicalVolume.World0xc15cfc00x40f7000_PV.50a18baaf29b18fae8c1642927003ee3

$OPTICKS_PREFIX/lib/OKTest --compute 

