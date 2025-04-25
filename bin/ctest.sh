#!/bin/bash
usage(){ cat << EOU
ctest.sh
=========

Usage::

    $OPTICKS_PREFIX/bin/ctest.sh

    ctest.sh info_copy_run_ana
    ctest.sh info
    ctest.sh copy
    ctest.sh run
    ctest.sh ana

As ctest doesnt currently work from a readonly dir
this copies the released ctest metadata folders to TTMP
directory and runs the ctest from there

EOU
}


do_ctest()
{
   echo [ $FUNCNAME
   date
   ctest -N
   ctest $* --interactive-debug-mode 0 --output-on-failure 2>&1
   date
   echo ] $FUNCNAME
}


defarg="info_copy_run_ana"
arg=${1:-$defarg}

tdir=$OPTICKS_PREFIX/tests
tmp=/tmp/$USER/opticks
TMP=${TMP:-$tmp}
TTMP=$TMP/tests
log=ctest.log

vv="BASH_SOURCE arg defarg OPTICKS_PREFIX tdir tmp TMP TTMP log"


check_tdir()
{
    if [ ! -d "$tdir" ]; then
        echo $BASH_SOURCE - tdir $tdir does not exist - EXIT
        exit 1
    fi
}


if [ "${arg/info}" != "$arg" ]; then
    for v in $vv ; do printf "%20s : %s\n" "$v" "${!v}" ; done
fi 

if [ "${arg/copy}" != "$arg" ]; then
    check_tdir 
    echo $BASH_SOURCE - copy ctest from $tdir to TTMP $TTMP
    mkdir -p $TTMP
    cp -r ${tdir}/. ${TTMP}/
fi 

if [ "${arg/run}" != "$arg" ]; then
    check_tdir 
    echo [ $BASH_SOURCE - run ctest with tee logging to $TTMP/$log
    cd $TTMP
    pwd      | tee    $log
    do_ctest | tee -a $log
    pwd      | tee -a $log
    echo ] $BASH_SOURCE - run ctest with tee logging to $TTMP/$log
fi 

if [ "${arg/ana}" != "$arg" ]; then
    echo [ $BASH_SOURCE - CTestLog.py $TTMP/$log
    which CTestLog.py
    CTestLog.py $TTMP/$log --slow 2.5,5,15
    echo ] $BASH_SOURCE - CTestLog.py $TTMP/$log
fi 

exit 0

