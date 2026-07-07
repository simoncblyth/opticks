#!/usr/bin/env bash
usage(){ cat << EOU
sreportdb.sh
==============

~/o/sysrap/tests/sreportdb.sh

* creates sqlite3 db, loads schema, populates using run.npy and evsmry.npy from *runfold*

EOU
}

name=sreportdb
bin=$name

DBNAME=$name.sqlite3


if [ -n "$SREPORT_ARCHIVE_DIR" ]; then
   mkdir -p $SREPORT_ARCHIVE_DIR
   DBFOLD=${SREPORT_ARCHIVE_DIR}
   INFOLD=${SREPORT_ARCHIVE_DIR}
else
   tmp=/tmp/$USER/opticks
   TMP=${TMP:-$tmp}
   FOLD=$TMP/$name
   mkdir -p $FOLD

   DBFOLD=$FOLD

   repfold=/data1/blyth/tmp/GEOM/J26_1_1_opticks_Debug/CSGOptiXSMTest/ALL1_Debug_Philox_medium_scan_first_sreport
   REPFOLD=${REPFOLD:-$repfold}

   archive=/tmp/blyth/opticks/sreport_archive_dir
   ARCHIVE=${ARCHIVE:-$archive}

   #INFOLD=$REPFOLD
   INFOLD=$ARCHIVE
fi

DBPATH=$DBFOLD/$DBNAME



defarg="info_run_check"
allarg="info_clean_run_check_query"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name bin tmp TMP FOLD DBFOLD DBNAME DBPATH REPFOLD ARCHIVE INFOLD defarg arg allarg SREPORT_ARCHIVE_DIR"

cd $(dirname $(realpath $BASH_SOURCE))

if [ "${arg/info}" != "$arg" ]; then
   for v in $vv ; do printf "%30s : %s\n" "$v" "${!v}" ; done
fi

if [ "${arg/clean}" != "$arg" ]; then
   rm -f $DBPATH
fi

if [ "${arg/run}" != "$arg" ]; then
   $bin $DBFOLD $INFOLD
   [ $? -ne 0 ] && echo $BASH_SOURCE - run error && exit 2
fi

if [ "${arg/check}" != "$arg" ]; then
   echo "select * from opticks_versionset   ;" | sqlite3 -table $DBPATH
   [ $? -ne 0 ] && echo $BASH_SOURCE - check error && exit 2
   echo "select * from opticks_runs   ;" | sqlite3 -table $DBPATH
   [ $? -ne 0 ] && echo $BASH_SOURCE - check error && exit 2
   echo "select * from opticks_events ;" | sqlite3 -table $DBPATH
   [ $? -ne 0 ] && echo $BASH_SOURCE - check error && exit 2
fi

if [ "${arg/query}" != "$arg" ]; then
   sqlite3 $DBPATH
fi

