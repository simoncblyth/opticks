#!/bin/bash -l 

bin=CG4Test 

export CManager=INFO
export CEventAction=INFO
export CGenstepCollector=INFO
export OpticksGenstep=INFO

if [ "$(uname)" == "Linux" ]; then 
    gdb -ex r --args $bin
else
    lldb__ $bin 
fi 

