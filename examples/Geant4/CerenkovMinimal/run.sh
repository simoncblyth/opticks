#!/bin/bash -l

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

which CerenkovMinimal

#export CKM_OPTICKS_EXTRA="--dbgrec"

export G4Opticks=INFO
export CManager=INFO


if [ "$(uname)" == "Darwin" ]; then 
    lldb__ CerenkovMinimal $* 
else
    CerenkovMinimal $*
fi

