#!/bin/bash -l

opticks-
opticks-id
opticks-boost-info


sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/$name/build 

rm   -rf $bdir && mkdir -p $bdir && cd $bdir && pwd 


if [ -z "$JUNOTOP" ]; then 
    echo no JUNOTOP
else
    source $JUNOTOP/bashrc.sh
    if [ ! -d "$JUNO_EXTLIB_Boost_HOME" ]; then 
       echo missing JUNO_EXTLIB_Boost_HOME
       exit 1 
    fi
    env | grep JUNO_EXTLIB_Boost

    echo $CMAKE_PREFIX_PATH | tr ":" "\n" 
fi


export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH}:$(opticks-prefix)/externals 


cmake $sdir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_INSTALL_PREFIX=$(opticks-prefix) \
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules 


cat << EON > /dev/null
Need to know basis, the below confuses finding boost

    -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
    -DBOOST_LIBRARYDIR=$(opticks-boost-libdir) \
    -DBoost_NO_SYSTEM_PATHS=1 

    -DBoost_NO_BOOST_CMAKE=ON

    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \

EON



make
make install   

exe=$(opticks-prefix)/lib/$name

if [ "$(uname)" == "Linux" ]; then
   ldd $exe
fi 



if [ -f "$exe" ]; then
    ls -l $exe
    echo running installed exe $exe
    $exe
else 
    echo failed to install exe to $exe 
fi 



cat << EON > /dev/null

    [blyth@localhost UseUseBoost]$ find /home/blyth/junotop/ExternalLibs/Boost/ -name '*onfig.cmake'
    ...
    /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_thread-1.70.0/boost_thread-config.cmake
    /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0/BoostConfig.cmake
    /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_headers-1.70.0/boost_headers-config.cmake
    /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_exception-1.70.0/boost_exception-config.cmake
    [blyth@localhost UseUseBoost]$ 


* http://cmake.3232098.n2.nabble.com/Boost-1-70-0-FindBoost-issues-No-linker-libraries-found-td7599380.html

Installation of Boost 1.70 generates and deploys a brand new BoostConfig.cmake 
machinery which is still being tested. 

Try setting -DBoost_NO_BOOST_CMAKE=ON to get the old FindBoost.cmake behaviour 
https://github.com/Kitware/CMake/blob/master/Modules/FindBoost.cmake#L238







EON

