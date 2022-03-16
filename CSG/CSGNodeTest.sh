#!/bin/bash -l 

default=remote
arg=${1:-$default}

opticks-switch-key $arg 

bin=CSGNodeTest 

if [ -n "$DEBUG" ]; then 
   if [ "$(uname)" == "Darwin" ]; then
       lldb__ $bin
   else
       gdb $bin
   fi
else
   $bin
fi 


