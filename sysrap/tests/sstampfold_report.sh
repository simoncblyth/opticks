#!/bin/bash -l 
usage(){ cat << EOU
sstampfold_report.sh
=======================

::

   ~/opticks/sysrap/tests/sstampfold_report.sh

   PICK=AB ~/opticks/sysrap/tests/sstampfold_report.sh ana

   PICK=A TLIM=-5,500 ~/opticks/sysrap/tests/sstampfold_report.sh ana


NB the sstampfold_report executable can be used without using this
script by invoking the executable from appropriate directories.
For example::

   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/ALL/

Which contains NPFold directories with names p001 n001 etc..
that match the fold prefixed hardcoded into the sstampfold_report 
executable. 

The default output FOLD when no envvar is defined is "../sstampfold_report" 
relative to the invoking directory of directory argument, eg::

   /data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/CSGOptiXSMTest/sstampfold_report/

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=sstampfold_report
script=$SDIR/$name.py
bin=$name


source $HOME/.opticks/GEOM/GEOM.sh 


job=N1
JOB=${JOB:-$job}

dir=unknown 
case $JOB in 
  L1) dir=/hpcfs/juno/junogpu/blyth/tmp/GEOM/$GEOM/jok-tds/ALL0 ;;
  N1) dir=/data/blyth/opticks/GEOM/$GEOM/jok-tds/ALL0 ;;
  N2) dir=/data/blyth/opticks/GEOM/$GEOM/G4CXTest/ALL0 ;;
  N3) dir=/data/blyth/opticks/GEOM/$GEOM/CSGOptiXSMTest/ALL ;;
esac

cd $dir
[ $? -ne 0 ] && echo $BASH_SOURCE : NO SUCH DIRECTORY job $job dir $dir && exit 0 


export FOLD=$PWD/../$name   ## set FOLD used by binary, for ana 
export MODE=2               ## 2:matplotlib plotting 


defarg="run_info_ana"
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
    echo run $bin
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi

if [ "${arg/noa}" != "$arg" ]; then 
    echo $BASH_SOURCE : noa exit 
    exit 0
fi

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

exit 0 

