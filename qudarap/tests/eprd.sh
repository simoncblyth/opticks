#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
eprd-desc(){ cat << EOD
$BASH_SOURCE:$FUNCNAME
=========================

    TEST              : $TEST 

    EPRD_NRMT         : $EPRD_NRMT 
    EPRD_FLAG         : $EPRD_FLAG 

EOD
}

eprd-set(){
    local nrmt=$1
    local flag=$2
    export EPRD_NRMT=${EPRD_NRMT:-$nrmt}
    export EPRD_FLAG=${EPRD_FLAG:-$flag}
}
eprd-unset(){
    unset EPRD_NRMT
    unset EPRD_FLAG
}


eprd-set 0,0,1,100 101,0,0,10 

if [ -n "$VERBOSE" ]; then 
    eprd-desc
fi
    




