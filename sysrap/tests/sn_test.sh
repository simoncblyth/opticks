#!/bin/bash -l 
usage(){ cat << EOU
sn_test.sh
==========

~/o/sysrap/tests/sn_test.sh 


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))
name=sn_test

export FOLD=/tmp/$name
mkdir -p $FOLD

bin=$FOLD/$name
script=$name.py 
 
#defarg="info_build_run_ana"
defarg="info_build_run"
arg=${1:-$defarg}

opt=-DWITH_CHILD
export s_pool_level=2

vars="BASH_SOURCE bin script opt"

if [ "${arg/info}" != "$arg" ]; then 
    for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc \
        ../sn.cc \
        ../s_tv.cc \
        ../s_pa.cc \
        ../s_bb.cc \
        ../s_csg.cc \
        -I.. \
        -I$HOME/np \
        -I$OPTICKS_PREFIX/externals/glm/glm \
        -lm \
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

