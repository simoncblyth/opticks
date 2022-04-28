#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

cmake-local() 
{
    local sdir=$1;
    local bdir=$PWD;
    [ "$sdir" == "$bdir" ] && echo ERROR sdir and bdir are the same $sdir && return 1000;
    local rc;
    cmake $sdir -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Debug \
        -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
        -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
        -DCMAKE_MODULE_PATH=$OPTICKS_HOME/cmake/Modules 
    rc=$?;
    return $rc
}

cmake-local $sdir 
[ $? -ne 0 ] && echo $msg cmake error && exit 1

make
[ $? -ne 0 ] && echo $msg make error && exit 2

make install   
[ $? -ne 0 ] && echo $msg install error && exit 3

which $name && $name
[ $? -ne 0 ] && echo $msg run error && exit 4

exit 0 
