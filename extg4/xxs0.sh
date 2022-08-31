#!/bin/bash -l
usage(){ cat << EOU
xxs0.sh : reboot of xxs.sh using X4Intersect::Scan
======================================================

EOU
}


export GRIDSCALE=0.1  ## HMM need to align the defaults used by gxt.sh and xxs0.sh 




arg=${1:-run_ana}

bin=X4IntersectSolidTest
log=$bin.log

source $(dirname $BASH_SOURCE)/../bin/COMMON.sh

if [ "${arg/info}" != "$arg" ]; then 
    vars="BASH_SOURCE arg bin GEOM"
    for var in $vars ; do printf "%30s : %s \n" $var ${!var} ; done 
fi

if [ "${arg/run}" != "$arg" ]; then
    [ -f "$log" ] && rm $log 
    $bin
    [ $? -ne 0 ] && echo $BASH_SOURCE run error && exit 1 
fi  

if [ "${arg/dbg}" != "$arg" ]; then
    case $(uname) in
        Darwin) lldb__ $bin ;;
        Linux)  gdb__ $bin ;;
    esac
    [ $? -ne 0 ] && echo $BASH_SOURCE dbg error && exit 2
fi  

if [ "${arg/ana}"  != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i $(dirname $BASH_SOURCE)/tests/X4IntersectSolidTest.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana interactive error && exit 3
fi 

exit 0 
