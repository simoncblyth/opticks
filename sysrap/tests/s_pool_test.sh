#!/bin/bash

usage(){ cat << EOU

~/o/sysrap/tests/s_pool_test.sh 


EOU
}


cd $(dirname $(realpath $BASH_SOURCE))

name=s_pool_test 
bin=/tmp/$name



gdb__ () 
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



export s_pool_level=1

#test=vector_0_FAILS # double dtor fail
#test=vector_1_FAILS # double dtor fail
#test=vector_2 # OK for pointers
test=create_delete_0 # OK for pointers

export TEST=${TEST:-$test}


defarg="info_build_run"
arg=${1:-$defarg}


vars="BASH_SOURCE name bin arg test TEST"

if [ "${arg/info}" != "$arg" ]; then
    for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi

if [ "${arg/build}" != "$arg" ]; then
    gcc $name.cc Obj.cc -g -std=c++11 -lstdc++ -I.. -o $bin
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

exit 0 


