#!/bin/bash -l 

post=0,0,0,0
momw_polw=0,0,-1,1,0,1,0,500
flag=0,0,0,0

export POST=${POST:-$post}
export MOMW_POLW=${MOMW_POLW:-$momw_polw}
export FLAG=${FLAG:-$flag}


ephoton-desc(){ cat << EOD
$BASH_SOURCE:$FUNCNAME
=========================

    POST      : $POST 
    MOMW_POLW : $MOMW_POLW
    FLAG      : $FLAG 


EOD
}

ephoton-desc


