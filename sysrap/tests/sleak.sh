#!/bin/bash -l 

usage(){ cat << EOU
sleak.sh 
==========

EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))

name=sleak
src=$SDIR/$name.cc
script=$SDIR/$name.py
bin=${TMP:-/tmp/$USER/opticks}/$name/$name    ## standalone binary
defarg="info_build_run_ana_info"
arg=${1:-$defarg}


DIR=/data/blyth/opticks/GEOM/J23_1_0_rc3_ok0/jok-tds/ALL0
export SLEAK_FOLD=${DIR}_${name}   ## SLEAK_FOLD is output directory used by binary and analysis

vars="0 BASH_SOURCE SDIR DIR SLEAK_FOLD"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%25s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
    mkdir -p $(dirname $bin)
    gcc $src -g -std=c++11 -lstdc++ -I$SDIR/.. -o $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE : build error && exit 1
fi


if [ "${arg/dbg}" != "$arg" ]; then 
    cd $DIR
    [ $? -ne 0 ] && echo $BASH_SOURCE : NO SUCH DIRECTORY && exit 0 

    dbg__ $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg error && exit 3
fi

if [ "${arg/run}" != "$arg" ]; then 
    cd $DIR
    [ $? -ne 0 ] && echo $BASH_SOURCE : NO SUCH DIRECTORY : JOB  $JOB DIR $DIR && exit 0 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run error && exit 3
fi


if [ "${arg/ana}" != "$arg" ]; then 
    export COMMANDLINE="~/o/sysrap/tests/sleak.sh"
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE : ana error && exit 3
fi


