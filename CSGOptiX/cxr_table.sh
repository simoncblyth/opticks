#!/bin/bash -l 

msg="=== $FUNCNAME : "

arg=${1:-tab} 

if [ "$arg" == "tab" -o "$arg" == "tab_water" ]; then
    ./cxr_grab.sh $arg --rst
else
    echo $msg unexpected arg $arg 
fi 




