#!/bin/bash -l 
if [ "$(uname)" == "Darwin" ]; then 
    GDB=lldb__ ./cxs.sh 
else
    GDB=gdb ./cxs.sh 
fi
