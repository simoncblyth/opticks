#!/bin/bash -l
usage(){ cat << EOU
ct_chk.sh
===========

Normally environment setup us mostly done by ../bin/GEOM_.sh 
but here doing it manually as check of what depends on what.

EOU
}

arg=${1:-run}
bin=CSGSimtraceTest

loglevels()
{
   export CSGSimtrace=INFO
   export CSGFoundry=INFO
}
loglevels


export GEOM=nmskSolidMask__U1   # needed by both run and ana 


if [ "$arg" == "run" ]; then

    export ${GEOM}_CFBaseFromGEOM=/tmp/$USER/opticks/GEOM/$GEOM

    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi

if [ "$arg" == "ana" ]; then

    export FOLD=/tmp/$USER/opticks/GEOM/$GEOM/$bin/ALL

    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/$bin.py
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 

