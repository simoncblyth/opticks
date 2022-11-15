#!/bin/bash -l 

export Local_G4Cerenkov_modified_DISABLE=1
export Local_DsG4Scintillation_DISABLE=1
export U4Recorder=INFO
export GEOM=hamaLogicalPMT
export U4RecorderTest__PRIMARY_MODE=iphoton

bin=U4PMTFastSimTest

defarg="dbg"
arg=${1:-$defarg}

if [ "$arg" == "run" ]; then
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi 

if [ "$arg" == "dbg" ]; then
    case $(uname) in 
        Darwin) lldb__ $bin ;;
        Linux)   gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi 

exit 0 

