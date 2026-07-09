#!/usr/bin/env bash
usage(){ cat << EOU
sreportdb.sh
==============

~/o/sysrap/tests/sreportdb.sh

1. creates sreportdb.sqlite3
2. loads schema sql defining tables : opticks_versionset, opticks_runs, opticks_events
3. populates the tables from report folders containing run.npy and evsmry.npy arrays


info
   dump variables

clean
   delete DBPATH

run
   invoke sreportdb with DBFOLD and INFOLD arguments

check
   invoke few simple queries against DBPATH using sqlite3 tool

query
   interactive use of sqlite3 commandline tool

ls
   list the configured DBPATH

report
   requires sudoer privilege - acting as gitlab-runner user recreate the gitlab-ci reports database


EOU
}

name=sreportdb
bin=$name

DBNAME=$name.sqlite3

#export sreportdb__dupe_import=1  ## uncomment to test dupe prevention
#export SREPORT_ARCHIVE_DIR=/tmp/blyth/opticks/sreport_archive_dir


if [ -n "$SREPORT_ARCHIVE_DIR" ]; then

   printf "$BASH_SOURCE - standard route used by ~/oj/.gitlab-ci.yml report job"
   mkdir -p $SREPORT_ARCHIVE_DIR
   DBFOLD=${SREPORT_ARCHIVE_DIR}
   INFOLD=${SREPORT_ARCHIVE_DIR}

   printf "$BASH_SOURCE - import from SREPORT_ARCHIVE_DIR : $SREPORT_ARCHIVE_DIR \n"
else

   printf "$BASH_SOURCE - development route used to check DB creation machinery"

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

   printf "$BASH_SOURCE - test import from INFOLD $INFOLD \n"

fi

DBPATH=$DBFOLD/$DBNAME



defarg="info_run_check"
allarg="info_clean_run_check_query_ls_report"
arg=${1:-$defarg}

vv="BASH_SOURCE PWD name bin tmp TMP FOLD DBFOLD DBNAME DBPATH REPFOLD ARCHIVE INFOLD defarg arg allarg SREPORT_ARCHIVE_DIR sreportdb__dupe_import"

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
   sqlite3 -readonly -box $DBPATH
fi

if [ "${arg/ls}" != "$arg" ]; then
   ls -alst $DBPATH
fi

if [[ "$arg" == "report" ]]; then
    BIN=$(which sreportdb)
    DIR=/data1/gitlab-runner/sreport_archive
    echo $BASH_SOURCE - ACT LIKE THE GITLAB-CI REPORT STAGE - RECREATING DB FROM REPORTS
    sudo -u gitlab-runner bash -c "$BIN $DIR $DIR"
fi

exit 0

