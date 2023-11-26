#!/bin/bash -l 
usage(){ cat << EOU
sprof_fold_report.sh
=======================

::

   ~/opticks/sysrap/tests/sprof_fold_report.sh


NB the sprof_fold_report executable can be used without using this
script by invoking the executable from appropriate directories.
For example::

   cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL
   sprof_fold_report
   ls ../sprof_fold_report/
   
An appropriate directory contains NPFold sub-directories 
with names p001 n001 etc.. that match the fold prefix strings 
hardcoded into the sprof_fold_report executable. 

The default output FOLD when no envvar is defined is "../sprof_fold_report" 
relative to the invoking directory of directory argument.

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=sprof_fold_report
script=$SDIR/$name.py
bin=$name

##L
#cd /hpcfs/juno/junogpu/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/p001
#cd /hpcfs/juno/junogpu/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0
#cd /hpcfs/juno/junogpu/blyth/tmp/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0/n010
##N
cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0
#cd /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL

[ $? -ne 0 ] && echo $BASH_SOURCE : NO SUCH DIRECTORY && exit 0 


export FOLD=$PWD/../$name   ## set FOLD to the default used by the binary, for ana 
export MODE=2               ## 2:matplotlib plotting 


defarg="run_info_ana"
#defarg="run_info"
arg=${1:-$defarg}

vars="0 BASH_SOURCE SDIR FOLD PWD name bin script"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

exit 0 

