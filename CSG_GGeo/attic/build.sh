#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
sdir=$(pwd)
name=$(basename $sdir) 

bdir=/tmp/$USER/opticks/${name}.build 
rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 

[ -z "$OPTICKS_PREFIX" ] && echo $msg MISSING OPTICKS_PREFIX && exit 1
[ -z "$OPTICKS_HOME" ]   && echo $msg MISSING OPTICKS_HOME   && exit 2

cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DOPTICKS_PREFIX=${OPTICKS_PREFIX} \
    -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
    -DCMAKE_INSTALL_PREFIX=${OPTICKS_PREFIX}

[ $? -ne 0 ] && echo $msg : config error && exit 1 
     
make
[ $? -ne 0 ] && echo $msg : make error && exit 2 

make install   
[ $? -ne 0 ] && echo $msg : install error && exit 3


exit 0 

