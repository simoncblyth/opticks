#!/bin/bash -l 
usage(){ cat << EOU
cxr_resolution_scan.sh
=======================


EOU
}


source resolut.bash 

cxr_resolution_scan(){

    local msg="=== $FUNCNAME : "
    local factors="1 2 4 8 16"
    local factor
    for factor in $factors 
    do  
        local sz=$(resolut-size $factor)
        local px=$(resolut-pixels $sz)
        local mpx=$(resolut-mpixels $sz)
        printf "$msg factor %5d sz %15s px %15s mpx %15s \n" $factor $sz $px $mpx 

    

    done 
}

cxr_resolution_scan







