#!/bin/bash -l

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

which CerenkovMinimal

#export CKM_OPTICKS_EXTRA="--dbgrec"
export CKM_OPTICKS_EXTRA="--managermode 2"

log(){ cat << EOV | grep -v ^#
G4Opticks
#G4OpticksRecorder
#CManager
#CRecorder
CWriter
#CTrackInfo
#CG4Ctx
#OpticksRun
#OpticksEvent
#CG4
EOV
}


# evar control 
log_ls(){     for var in $(${VNAME:-log}) ; do printf "%20s : [%s] \n"  $var ${!var} ; done ; }
log_export(){ for var in $(${VNAME:-log}) ; do export $var=INFO                      ; done ; log_ls ; }  
log_unset(){  for var in $(${VNAME:-log}) ; do unset $var                            ; done ; log_ls ; }

log_export


if [ "$(uname)" == "Darwin" ]; then 
    lldb__ CerenkovMinimal $* 
else
    CerenkovMinimal $*
fi

