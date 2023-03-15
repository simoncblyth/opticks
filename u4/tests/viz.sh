#!/bin/bash -l 

DIR=$(dirname $BASH_SOURCE)

defarg="ana"
arg=${1:-$defarg}
script=$DIR/U4SimtraceTest.sh

if [ -n "$APID" -a -n "$BPID" ]; then 
    N=1 APID=$APID BPID=$BPID $script $arg
elif [ -n "$APID" ]; then
    N=0 APID=$APID AOPT=idx $script $arg
elif [ -n "$BPID" ]; then
    N=1 BPID=$BPID BOPT=idx $script $arg
else
    N=${N:-1} $script $arg
fi 






