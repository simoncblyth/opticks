#!/bin/bash
usage(){ cat << EOU

::

   ~/opticks/sysrap/tests/sstr_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
mkdir -p $TMP

name=sstr_test
bin=$TMP/$name


#test=prefix_suffix
#test=ParseInt
test=ExtractSize

export TEST=${TEST:-$test}

defarg="info_build_run"
arg=${1:-$defarg}

vv="BASH_SOURCE defarg arg TEST name bin TMP"

if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/build}" != "$arg" ]; then
   cc $name.cc -g -std=c++17 -lstdc++ -I.. -o $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2
fi

if [ "${arg/dbg}" != "$arg" ]; then
   source dbg__.sh
   dbg__ $bin
   [ $? -ne 0 ] && echo $BASH_SOURCE dbg  error && exit 3
fi

exit 0



