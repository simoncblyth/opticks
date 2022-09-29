#!/bin/bash -l 

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

defarg="info_config_build_install" 
arg=${1:-$defarg}

if [ "${arg/info}" != "$arg" ]; then
   vars="sdir name bdir arg"
   for var in $vars ; do printf "%20s : %20s \n" $var ${!var} ; done
fi 

if [ "${arg/config}" != "$arg" ]; then

    rm -rf $bdir && mkdir -p $bdir && cd $bdir && pwd

    cmake $sdir \
       -DCMAKE_BUILD_TYPE=Debug \
       -DOPTICKS_PREFIX=$OPTICKS_PREFIX \
       -DCMAKE_INSTALL_PREFIX=$OPTICKS_PREFIX \
       -DCMAKE_MODULE_PATH=$HOME/opticks/cmake/Modules

    [ $? -ne 0 ] && echo $BASH_SOURCE cmake error && exit 1 
fi 

if [ "${arg/build}" != "$arg" ]; then
   cmake --build   . 
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 2
fi

if [ "${arg/install}" != "$arg" ]; then
   cmake --install . 
   [ $? -ne 0 ] && echo $BASH_SOURCE install error && exit 3
fi 

exit 0 


