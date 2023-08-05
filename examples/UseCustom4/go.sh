#!/bin/bash -l
usage(){ cat << EOU
UseCustom4/go.sh 
==================

Testing CMake find_package with Custom4

To remove all installed Custom4 libs and headers::

   rm -rf /usr/local/opticks/lib/Custom4*
   rm -rf /usr/local/opticks/include/Custom4

EOU
}


sdir=$(cd $(dirname $BASH_SOURCE) && pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 
idir=/tmp/$USER/opticks/$name/install
bin=$idir/lib/$name
vars="sdir name bdir idir bin"

defarg="info_build_run"
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then 
   echo $CMAKE_PREFIX_PATH | tr ":" "\n"
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 

    rm -rf $idir && mkdir -p $idir && cd $idir && pwd 
    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

    cmake $sdir -DCMAKE_BUILD_TYPE=Debug -DCMAKE_INSTALL_PREFIX=$idir 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build/cmake error && exit 1 
    make
    [ $? -ne 0 ] && echo $BASH_SOURCE : build/make error && exit 1 
    make install   
    [ $? -ne 0 ] && echo $BASH_SOURCE : build/install error && exit 1 
fi 


if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 2
fi 

exit 0 

