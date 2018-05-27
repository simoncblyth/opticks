#!/bin/bash -l

opticks-

sdir=$(pwd)
name=$(basename $sdir)

bdir=$(opticks-prefix)/build/$name

if [ "$1" == "clean" ]; then
   echo $0 $1 cleaning bdir $bdir
   rm -rf $bdir 
fi 
mkdir -p $bdir && cd $bdir && pwd 

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules

make

if [ "$(uname)" == "Darwin" ]; then
   echo kludge 2s sleep 
   sleep 2 
fi

make install   

opticks-t $bdir


notes(){ cat << EON

The kludge sleep avoids::

    error: /opt/local/bin/install_name_tool: no LC_RPATH load command with path: /usr/local/opticks-cmake-overhaul/build/cudarap found in: /usr/local/opticks-cmake-overhaul/lib/LaunchSequenceTest (for architecture x86_64), required for specified option "-delete_rpath /usr/local/opticks-cmake-overhaul/build/cudarap"

EON
}

