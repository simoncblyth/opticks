#!/bin/bash -l 

export CRandomEngine=INFO

if [ "$(uname)" == "Darwin" ]; then
   lldb__ CRandomEngineTest $*
else
   gdb CRandomEngineTest $*
fi 


