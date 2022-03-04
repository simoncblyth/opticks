#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

export Frame=INFO
which OTracerTest 

source $OPTICKS_HOME/bin/geocache_hookup.sh 

if [ -n "$DEBUG" ]; then
    lldb__ OTracerTest 
else
    OTracerTest 
fi 

