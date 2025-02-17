#!/bin/bash 

usage(){ cat << EOU

  ~/opticks/sysrap/tests/s_locale_test.sh


epsilon:tests blyth$ LC_ALL=C  ~/opticks/sysrap/tests/s_locale_test.sh run
1000000000
1000000000
epsilon:tests blyth$ LC_ALL=POSIX  ~/opticks/sysrap/tests/s_locale_test.sh run
1000000000
1000000000
epsilon:tests blyth$ LC_ALL=en_US  ~/opticks/sysrap/tests/s_locale_test.sh run
1,000,000,000
1,000,000,000
epsilon:tests blyth$ LC_ALL=en_GB  ~/opticks/sysrap/tests/s_locale_test.sh run
1,000,000,000
1,000,000,000
epsilon:tests blyth$ LC_ALL=en_GB.UTF-8  ~/opticks/sysrap/tests/s_locale_test.sh run
1,000,000,000
1,000,000,000
epsilon:tests blyth$ 


Background info on locale

* https://www.ibm.com/docs/en/aix/7.1?topic=locales-understanding-locale-environment-variables



EOU
}


defarg="info_build_run"
arg=${1:-$defarg}

name=s_locale_test 
FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name


vars="0 defarg arg name FOLD bin" 

if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
          -std=c++11 -lstdc++ \
          -I.. \
          -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE compile error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

exit 0 


