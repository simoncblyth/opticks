#!/bin/bash  
usage(){ cat << EOU
U4RandomTest.sh
=================

::

   ./U4RandomTest.sh

EOU
}

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



#seqdir="/tmp/$USER/opticks/QSimTest/rng_sequence/rng_sequence_f_ni1000000_nj16_nk16_tranche100000"
#export OPTICKS_RANDOM_SEQPATH=$seqdir
#export OPTICKS_RANDOM_SEQPATH=$seqdir/rng_sequence_f_ni100000_nj16_nk16_ioffset000000.npy 

name=U4RandomTest 
export U4Random=INFO

defarg=dbg
arg=${1:-$defarg}


if [ "${arg/run}" != "$arg" ]; then 
    $name
    [ $? -ne 0 ] && echo run FAIL && exit 1
fi 

if [ "${arg/dbg}" != "$arg" ]; then 
    gdb__ $name
    [ $? -ne 0 ] && echo dbg FAIL && exit 2
fi 


exit 0 


