#!/bin/bash -l 
usage(){ cat << EOU
examples/UseFindOpticks/go.sh
===============================

Investigate mis-behaviour of PLOG LOG(LEVEL) 
logging from external libs that use Opticks.::

    ./go.sh info
    ./go.sh config
    ./go.sh build
    ./go.sh install

    ./go.sh info_config_build_install  # this is default 
    ./go.sh 

    ./go.sh run0
    ./go.sh run1


DemoLib envvar controls LOG(LEVEL) logging::

    epsilon:UseFindOpticks blyth$ ./go.sh run0
    PLOG::EnvLevel adjusting loglevel by envvar   key DemoLib level INFO fallback DEBUG
    2022-09-30 15:16:06.923 ERROR [16930060] [main@106] [DemoLibTest
    2022-09-30 15:16:06.924 ERROR [16930060] [DemoLib::Dump@8] [ before LOG(LEVEL) 
    2022-09-30 15:16:06.924 INFO  [16930060] [DemoLib::Dump@9] DemoLib::Dump
    2022-09-30 15:16:06.924 ERROR [16930060] [DemoLib::Dump@10] ] after LOG(LEVEL) 
    DemoLib::Dump
    2022-09-30 15:16:06.924 ERROR [16930060] [main@108] ]DemoLibTest


    epsilon:UseFindOpticks blyth$ ./go.sh run1
    2022-09-30 15:16:11.142 ERROR [16930309] [main@106] [DemoLibTest
    2022-09-30 15:16:11.143 ERROR [16930309] [DemoLib::Dump@8] [ before LOG(LEVEL) 
    2022-09-30 15:16:11.143 ERROR [16930309] [DemoLib::Dump@10] ] after LOG(LEVEL) 
    DemoLib::Dump
    2022-09-30 15:16:11.143 ERROR [16930309] [main@108] ]DemoLibTest
    epsilon:UseFindOpticks blyth$ 


EOU
}

sdir=$(pwd)
name=$(basename $sdir)
bdir=/tmp/$USER/opticks/$name/build 

defarg="info_config_build_install_run" 
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
   cd $bdir
   pwd
   cmake --build   . 
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 2
fi

if [ "${arg/install}" != "$arg" ]; then
   cd $bdir
   pwd
   cmake --install . 
   [ $? -ne 0 ] && echo $BASH_SOURCE install error && exit 3
fi 

if [ "$arg" == "run0" ]; then
   export DemoLib=INFO
   DemoLibTest  
fi 

if [ "$arg" == "run1" ]; then
   unset DemoLib
   DemoLibTest  
fi 



exit 0 

