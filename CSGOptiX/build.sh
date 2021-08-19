#!/bin/bash -l 

ver=${1:-6}
[ "$(uname)" == "Darwin" ] && ver=5

case $ver in
5) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX5_PREFIX} ;;
6) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
7) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX7_PREFIX} ;;
*) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
esac

echo ver $ver OPTICKS_OPTIX_PREFIX ${OPTICKS_OPTIX_PREFIX}

msg="=== $BASH_SOURCE :"
sdir=$(pwd)
name=$(basename $sdir)

chkvar()
{
    local msg="=== $FUNCNAME :"
    local var ; 
    for var in $* ; do 
        if [ -z "${!var}" -o ! -d "${!var}" ]; then 
            echo $msg missing required envvar $var ${!var} OR non-existing directory
            return 1
        fi
        printf "%20s : %s \n" $var ${!var}
    done
    return 0  
} 

chkvar OPTICKS_PREFIX OPTICKS_HOME OPTICKS_OPTIX_PREFIX
[ $? -ne 0 ] && echo $msg checkvar FAIL && exit 1


# Attempting to discern the OPTIX_VERSION by sourcing a 
# buildenv.sh script generated at config time is an 
# inherently flawed approach.
# Are now using the CSGOptiXVersion executable that is built and 
# installed together with the library, so can get the version in 
# scripts by capturing the output from that executable.

bdir=/tmp/$USER/opticks/${name}.build
rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOPTICKS_PREFIX=${OPTICKS_PREFIX} \
     -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
     -DCMAKE_INSTALL_PREFIX=${OPTICKS_PREFIX}

[ $? -ne 0 ] && echo $msg conf FAIL && exit 1


make
[ $? -ne 0 ] && echo $msg make FAIL && exit 2

make install   
[ $? -ne 0 ] && echo $msg install FAIL && exit 3


exit 0

