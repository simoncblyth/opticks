#!/bin/bash -l

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

which CerenkovMinimal

if [ "$(uname)" == "Darwin" ]; then 
    lldb_ CerenkovMinimal 
else
    CerenkovMinimal 
fi

