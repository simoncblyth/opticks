#!/bin/bash -l 
usage(){ cat << EOU
s_csg_test.sh
=============

EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=s_csg_test

FOLD=/tmp/$name
export FOLD
mkdir -p $FOLD


bin=$FOLD/$name
script=$SDIR/$name.py 

#defarg="info_build_run_ana"
defarg="info_build_run"
arg=${1:-$defarg}


base=/tmp/blyth/opticks/U4TreeCreateTest/stree/_csg
#base=/tmp/U4Polycone_test/_csg

export BASE=${BASE:-$base}
export s_csg_level=0

opt="-DWITH_CHILD"


vars="BASH_SOURCE SDIR name bin script arg FOLD BASE opt"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then
    gcc \
        $opt \
        $SDIR/$name.cc \
        $SDIR/../sn.cc \
        $SDIR/../s_tv.cc \
        $SDIR/../s_pa.cc \
        $SDIR/../s_bb.cc \
        $SDIR/../s_csg.cc \
        -I$SDIR/.. \
        -I$HOME/np \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        $opt -g -std=c++11 -lstdc++ -o $bin

    [ $? -ne 0 ] && echo $BASH_SOURCE : build fail && exit 1
fi 

if [ "${arg/run}" != "$arg" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE : run fail && exit 2
fi 

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
       Darwin) lldb__ $bin ;;
       Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE : dbg fail && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 
