#!/bin/bash -l

./check.sh 
[ $? -ne 0 ] && echo check failed && exit 1

which CerenkovMinimal

#export CKM_OPTICKS_EXTRA="--dbgrec"
#export CKM_OPTICKS_EXTRA="--managermode 2 --nogpu --print_enabled"
export CKM_OPTICKS_EXTRA="--managermode 2 --print_enabled"


# logging evar control 
log_(){ cat << EOV
#G4Opticks
#G4OpticksRecorder
#CManager
#CRecorder
CGenstepCollector
#CWriter
#CTrackInfo
#CCtx
#OpticksRun
#OpticksEvent
#CG4
EOV
}

log_on(){  log_ |  grep  -v ^#  ; }
log_off(){ log_ |  grep  ^#  | tr "#" " " ;  }
log_all(){ log_on ; log_off ; }

log_ls(){     for var in $(${VNAME:-log_all}) ; do printf "%20s : [%s] \n"  $var ${!var} ; done ; }
log_export(){ for var in $(${VNAME:-log_on})  ; do export $var=INFO                      ; done ; }  
log_unset(){  for var in $(${VNAME:-log_off}) ; do unset $var                            ; done ; }
log_up(){     log_export ; log_unset ; log_ls ; }  

log_up



if [ "$(uname)" == "Darwin" ]; then 
    lldb__ CerenkovMinimal $* 
else
    CerenkovMinimal $*
fi

