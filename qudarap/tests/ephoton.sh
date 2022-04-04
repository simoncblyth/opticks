#!/bin/bash -l 

ephoton-desc(){ cat << EOD
$BASH_SOURCE:$FUNCNAME
=========================

    TEST      : $TEST 
    POST      : $POST 
    MOMW_POLW : $MOMW_POLW
    FLAG      : $FLAG 


EOD
}

msg="=== $BASH_SOURCE :"

if [ "$TEST" == "propagate_at_boundary_normal_incidence" ]; then 

    post=0,0,0,0
    momw_polw=0,0,-1,1,0,1,0,500
    flag=0,0,0,0

    export POST=$post
    export MOMW_POLW=$momw_polw
    export FLAG=$flag
    ephoton-desc


elif [ "$TEST" == "propagate_at_boundary" ]; then 

    post=0,0,0,0
    #momw_polw=1,0,-1,1,0,1,0,500   # 45 degree incidence against the normal 0,0,1  : but with 1.5/1.0 indices tahat is TIR
    #momw_polw=1,0,1,1,0,1,0,500    # 45 degree incidence with the normal 0,0,1 : 

    #momw_polw=1,0,-2,1,0,1,0,500   # steeper incidence against the normal 0,0,1  to avoid TIR
    momw_polw=1,0,2,1,0,1,0,500   # steeper incidence with the normal 0,0,1  to avoid TIR

    flag=0,0,0,0

    export POST=$post
    export MOMW_POLW=$momw_polw
    export FLAG=$flag
    ephoton-desc


elif [ "$TEST" == "env" ]; then 

    post=0,0,0,0
    momw_polw=1,0,-1,1,0,1,0,500
    flag=0,0,0,0

    export POST=${POST:-$post}
    export MOMW_POLW=${MOMW_POLW:-$momw_polw}
    export FLAG=${FLAG:-$flag}
    ephoton-desc


elif [ "$TEST" == "reflect_specular" ]; then 

    post=0,0,0,0
    momw_polw=1,0,-1,1,0,1,0,500
    flag=0,0,0,0

    export POST=${POST:-$post}
    export MOMW_POLW=${MOMW_POLW:-$momw_polw}
    export FLAG=${FLAG:-$flag}
    ephoton-desc

else
    unset POST
    unset MOMW_POLW 
    unset FLAG
    echo $msg TEST $TEST : unset environment : will use C++ defaults in quad4::ephoton for p0
fi 






