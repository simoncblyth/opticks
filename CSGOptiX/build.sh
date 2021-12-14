#!/bin/bash -l 

notes(){ cat << EON

Switching between OptiX versions is 
achieved by setting the OPTICKS_OPTIX_PREFIX envvar
which is used for example by cmake/Modules/FindOpticksOptiX.cmake

Hmm : how to generalize with am om7 ?

Attempting to discern the OPTIX_VERSION by sourcing a 
buildenv.sh script generated at config time is an 
inherently flawed approach.
Are now using the CSGOptiXVersion executable that is built and 
installed together with the library, so can get the version in 
scripts by capturing the output from that executable.


HMM : the below looks to be almost identical to what om would 
do anyhow other than the "export OPTICKS_OPTIX_PREFIX"
But the difficulty is that need to have different settings 
in different pkgs. 

EON
}

msg="=== $BASH_SOURCE :"

echo $msg dont use this, use opticks-build7 or b7 shortcut 
exit 1 



ver=${1:-6}
[ "$(uname)" == "Darwin" ] && ver=5

case $ver in
5) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX5_PREFIX} ;;
6) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
7) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX7_PREFIX} ;;
*) export OPTICKS_OPTIX_PREFIX=${OPTICKS_OPTIX6_PREFIX} ;;
esac

echo ver $ver OPTICKS_OPTIX_PREFIX ${OPTICKS_OPTIX_PREFIX}

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

