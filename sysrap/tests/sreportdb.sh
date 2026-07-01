#!/usr/bin/env bash

usage(){ cat << EOU

sreportdb.sh

* creates sqlite3 db, loads schema, populates using run.npy and evsmry.npy from *runfold*

EOU
}

name=sreportdb

bin=$name
db=/tmp/$name.sqlite3
runfold=/data1/blyth/tmp/GEOM/J26_1_1_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_first_sreport


defarg="info_run_check"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name bin db sql runfold defarg arg"

cd $(dirname $(realpath $BASH_SOURCE))

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/clean}" != "$arg" ]; then
   rm -f $db
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin $db $runfold
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2
fi

if [ "${arg/check}" != "$arg" ]; then
   echo "select * from opticks_events ;" | sqlite3 -table $db
   [ $? -ne 0 ] && echo $BASH_SOURCE - check error && exit 2
fi

if [ "${arg/query}" != "$arg" ]; then
   sqlite3 $db
fi


