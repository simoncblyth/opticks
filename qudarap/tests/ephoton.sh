#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
ephoton-desc(){ cat << EOD
$BASH_SOURCE:$FUNCNAME
=========================

    TEST              : $TEST 

    EPHOTON_POST      : $EPHOTON_POST 
    EPHOTON_MOMW_POLW : $EPHOTON_MOMW_POLW
    EPHOTON_FLAG      : $EPHOTON_FLAG 

EOD
}

ephoton-set(){
    local post=$1
    local momw_polw=$2
    local flag=$3

    export EPHOTON_POST=${EPHOTON_POST:-$post}
    export EPHOTON_MOMW_POLW=${EPHOTON_MOMW_POLW:-$momw_polw}
    export EPHOTON_FLAG=${EPHOTON_FLAG:-$flag}
}

ephoton-unset(){
    unset EPHOTON_POST 
    unset EPHOTON_MOMW_POLW
    unset EPHOTON_FLAG
}

ephoton-propagate_at_boundary_normal_incidence(){
    local post=0,0,0,0
    local momw_polw=0,0,-1,1,0,1,0,500
    local flag=0,0,0,0
    ephoton-set $post $momw_polw $flag 
}

ephoton-propagate_at_boundary(){
    local post=0,0,0,0
    #local momw_polw=1,0,-1,1,0,1,0,500   # 45 degree incidence against the normal 0,0,1  : but with 1.5/1.0 indices tahat is TIR
    #local momw_polw=1,0,1,1,0,1,0,500    # 45 degree incidence with the normal 0,0,1 : 

    #local momw_polw=1,0,-2,1,0,1,0,500   # steeper incidence against the normal 0,0,1  to avoid TIR
    local momw_polw=1,0,2,1,0,1,0,500   # steeper incidence with the normal 0,0,1  to avoid TIR

    local flag=0,0,0,0
    ephoton-set $post $momw_polw $flag 
}

ephoton-env(){
    local post=0,0,0,0
    local momw_polw=1,0,-1,1,0,1,0,500
    local flag=0,0,0,0
    ephoton-set $post $momw_polw $flag 
}

ephoton-reflect_specular(){
    local post=0,0,0,0
    local momw_polw=1,0,-1,1,0,1,0,500
    local flag=0,0,0,0
    ephoton-set $post $momw_polw $flag 
}

ephoton-mock_propagate(){
    local post=0,0,0,0
    local momw=1,0,-1,1
    local polw=0,1,0,500
    local flag=0,0,0,0
    ephoton-set $post ${momw},${polw} $flag 
}

ephoton-default(){
    ephoton-unset 
    echo $msg TEST $TEST : unset environment : will use C++ defaults in quad4::ephoton for p0
}

case $TEST in 
   propagate_at_boundary_normal_incidence) ephoton-propagate_at_boundary_normal_incidence ;;
   propagate_at_boundary)                  ephoton-propagate_at_boundary                  ;;
   env)                                    ephoton-env  ;;  
   reflect_specular)                       ephoton-reflect_specular  ;;
   mock_propagate*)                        ephoton-mock_propagate ;; 
                 *)                        ephoton-default ;;
esac


if [ -n "$VERBOSE" ]; then 
   ephoton-desc
fi 


