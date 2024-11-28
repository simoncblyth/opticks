#!/bin/bash
usage(){ cat << EOU

~/o/sysrap/tests/stra_test.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=stra_test 
export FOLD=/tmp/$name
mkdir -p $FOLD
bin=$FOLD/$name

defarg="info_build_run" 
arg=${1:-$defarg}
opt=-g

#test=Copy_Columns_3x4
test=Elements
export TEST=${TEST:-$test}

vars="BASH_SOURCE arg opt FOLD test TEST"



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



if [ "${arg/info}" != "$arg" ]; then 
   for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 
fi 

if [ "${arg/build}" != "$arg" ]; then 
   gcc $name.cc $opt -std=c++11 -lstdc++ -I.. -I$OPTICKS_PREFIX/externals/glm/glm -o $bin
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


