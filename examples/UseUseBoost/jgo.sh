#!/bin/bash -l

opticks-
opticks-id
opticks-boost-info

notes(){ cat << EON

The boost lib version from UseBoost must match that of UseUseBoost 
else get double free corruption crashes from having multiple boost libs
active.  

jgo.sh
    uses the JUNOTOP Boost libs
go.sh
    uses system Boost libs 

That means that jgo.sh from UseBoost needs to be followed by jgo.sh from UseUseBoost
to avoid the crash.

EON
}



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
    -DCMAKE_MODULE_PATH=$(opticks-home)/cmake/Modules \
    -DOPTICKS_PREFIX=$(opticks-prefix)

rc=$? && [ "$rc" != "0" ] && echo cmake RC $rc && exit $rc



cat << EON > /dev/null


Need to know basis, the below confuses finding boost

    -DBOOST_INCLUDEDIR=$(opticks-boost-includedir) \
    -DBOOST_LIBRARYDIR=$(opticks-boost-libdir) \
    -DBoost_NO_SYSTEM_PATHS=1 

    -DBoost_NO_BOOST_CMAKE=ON

    -DCMAKE_PREFIX_PATH=$(opticks-prefix)/externals \

EON



make
rc=$? && [ "$rc" != "0" ] && echo make RC $rc && exit $rc


make install   
rc=$? && [ "$rc" != "0" ] && echo install RC $rc && exit $rc

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



cat << EOX > /dev/null


Wierd CMake find boost mixup 


-- Detecting CXX compile features - done
-- Configuring UseUseBoost
-- Found Boost 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0
--   Requested configuration: QUIET REQUIRED COMPONENTS system;program_options;filesystem;regex
-- Found boost_headers 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_headers-1.70.0
-- Found boost_system 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_system-1.70.0
--   libboost_system.so.1.70.0
-- Adding boost_system dependencies: headers
-- Found boost_program_options 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_program_options-1.70.0
--   libboost_program_options.so.1.70.0
-- Adding boost_program_options dependencies: headers
-- Found boost_filesystem 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_filesystem-1.70.0
--   libboost_filesystem.so.1.70.0
-- Adding boost_filesystem dependencies: headers
-- Found boost_regex 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/boost_regex-1.70.0
--   libboost_regex.so.1.70.0
-- Adding boost_regex dependencies: headers
-- Found Boost: /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0/BoostConfig.cmake (found version "1.70.0") found components:  system program_options filesystem regex 
-- Found Boost 1.70.0 at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0
--   Requested configuration: QUIET REQUIRED COMPONENTS system;program_options;filesystem;regex;1.70.0
CMake Error at /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0/BoostConfig.cmake:95 (find_package):
  Could not find a package configuration file provided by "boost_1.70.0"
  (requested version 1.70.0) with any of the following names:

    boost_1.70.0Config.cmake
    boost_1.70.0-config.cmake

  Add the installation prefix of "boost_1.70.0" to CMAKE_PREFIX_PATH or set
  "boost_1.70.0_DIR" to a directory containing one of the above files.  If
  "boost_1.70.0" provides a separate development package or SDK, be sure it
  has been installed.
Call Stack (most recent call first):
  /home/blyth/junotop/ExternalLibs/Boost/1.70.0/lib/cmake/Boost-1.70.0/BoostConfig.cmake:124 (boost_find_dependency)
  /home/blyth/junotop/ExternalLibs/Cmake/3.15.2/share/cmake-3.15/Modules/FindBoost.cmake:422 (find_package)
  /home/blyth/junotop/ExternalLibs/Cmake/3.15.2/share/cmake-3.15/Modules/CMakeFindDependencyMacro.cmake:47 (find_package)
  /home/blyth/local/opticks/lib64/cmake/useboost/useboost-config.cmake:7 (find_dependency)
  CMakeLists.txt:25 (find_package)

EOX

