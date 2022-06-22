#!/bin/bash -l 
usage(){ cat << EOU
storch_test.sh
================

CPU test of CUDA code to generate torch photons using s_mock_curand.h::

   ./storch_test.sh build
   ./storch_test.sh run
   ./storch_test.sh ana
   ./storch_test.sh build_run_ana   # default 

EOU
}

msg="=== $BASH_SOURCE :"
name=storch_test 
fold=/tmp/$name
mkdir -p $fold

arg=${1:-build_run_ana}

if [ "${arg/build}" != "$arg" ]; then 
    gcc $name.cc -std=c++11 -lstdc++ \
        -DMOCK_CURAND \
        -I.. \
        -I/usr/local/cuda/include \
        -I$OPTICKS_PREFIX/externals/plog/include \
        -L$OPTICKS_PREFIX/lib \
        -lSysRap \
        -o $fold/$name 

    [ $? -ne 0 ] && echo $msg build error && exit 1 
fi 

if [ "${arg/run}" != "$arg" ]; then 
    $fold/$name
    [ $? -ne 0 ] && echo $msg run error && exit 2 
fi

if [ "${arg/ana}" != "$arg" ]; then 
    export FOLD=$fold
    echo $msg FOLD $FOLD
    ${IPYTHON:-ipython} --pdb -i $name.py 
    [ $? -ne 0 ] && echo $msg ana error && exit 3 
fi


if [ "${arg/grab}" != "$arg" ]; then 
    echo $msg fold $fold

    xdir=$fold/       ## require trailing slash to avoid rsync duplicating path element 
    from=P:$xdir
    to=$xdir

    vars="fold xdir from to"
    dumpvars(){ for var in $vars ; do printf "%-30s : %s \n" $var "${!var}" ; done ; } 
    dumpvars
    read -p "$msg Enter YES to proceed with rsync between from and to " ans 
    if [ "$ans" == "YES" ]; then 
        echo $msg proceeding 
        mkdir -p $to 
        rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
        ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
        ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `
    else
       echo $msg skipping
    fi  

fi


exit 0 


