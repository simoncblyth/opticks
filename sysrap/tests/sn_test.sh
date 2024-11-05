#!/bin/bash
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

cuda_prefix=/usr/local/cuda
CUDA_PREFIX=${CUDA_PREFIX:-$cuda_prefix}

#test=ALL
#test=create_0
#test=deepcopy_0
#test=list_tree_0
#test=difference_and_list_tree_0
test=CreateSmallerTreeWithListNode_0
export TEST=${TEST:-$test}

logging()
{
    type $FUNCNAME
    export sn__level=2 
    export s_pool_level=2
}
[ -n "$LOG" ] && logging 


gdb__() 
{ 
    if [ -z "$BP" ]; then
        H="";
        B="";
        T="-ex r";
    else
        H="-ex \"set breakpoint pending on\"";
        B="";
        for bp in $BP;
        do
            B="$B -ex \"break $bp\" ";
        done;
        T="-ex \"info break\" -ex r";
    fi;
    local runline="gdb $H $B $T --args $* ";
    echo $runline;
    date;
    eval $runline;
    date
}



vars="BASH_SOURCE bin script opt TEST sn__level"

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
        -I$CUDA_PREFIX/include \
        -lm \
        $opt -g -std=c++11 -lstdc++ -o $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 2 
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    gdb__ $bin 
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 3
fi 

if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $script
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 4
fi 

exit 0 

