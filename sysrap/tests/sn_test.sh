#!/bin/bash -l 
usage(){ cat << EOU
sn_test.sh
==========


EOU
}

SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)
name=sn_test

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$SDIR/$name.py 
 
defarg="info_build_run_ana"
arg=${1:-$defarg}

opt=-DWITH_CHILD
export s_pool_level=2

vars="BASH_SOURCE SDIR bin script opt"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $SDIR/$name.cc \
        $SDIR/../sn.cc \
        $SDIR/../s_tv.cc \
        $SDIR/../s_pa.cc \
        $SDIR/../s_bb.cc \
        $SDIR/../s_csg.cc \
        -I$SDIR/.. \
        -I$HOME/np \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        $opt -g -std=c++11 -lstdc++ -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    case $(uname) in 
    Linux)   gdb__ $bin ;;
    Darwin) lldb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

