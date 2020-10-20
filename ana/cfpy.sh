#!/bin/bash -l 

mode=${1:-0}

ana_cmd()
{
    local mode=${1:-0}
    local py=${2:-2.7}
    local fmt
    case $mode in 
      0) fmt="python%s $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show" ;;
      1) fmt="LV=box python%s histype.py" ;;
      2) fmt="python%s seq.py" ;;
    esac 
    local ana=$(printf "$fmt" $py);  
    echo "$ana" 
}

usage(){ cat << EON
To regenerate the event files read by these analysis scripts, run::

    LV=box GGeoTest=INFO tboolean.sh --generateoverride 10000 --noviz

EON
}

ana_log(){ 
    local mode=${1:-0}
    local py=${2:-2.7}
    local idx=${3:-0}
    echo /tmp/$USER/opticks/ana/ana_log_py${py}_mode${mode}_idx${idx}.log
}

cfpy()
{
    : compare output from running same cmd with two pythons
    local py2cmd="$(ana_cmd $mode 2.7)"
    local py3cmd="$(ana_cmd $mode 3)"

    local py2log="$(ana_log $mode 2.7)"
    local py3log="$(ana_log $mode 3)"

    mkdir -p $(dirname $py2log)

    echo "$py2cmd"
    eval "$py2cmd" 2> $py2log 1>&2
    [ $? -ne 0 ] && echo FAIL from py2.7 && cat $py2log && exit 1

    echo "$py3cmd"
    eval "$py3cmd" 2> $py3log 1>&2
    [ $? -ne 0 ] && echo FAIL from py3 && cat $py3log && exit 1


    local cmd="diff $py2log $py3log"
    echo $cmd
    eval $cmd
    echo rc $?
    echo $cmd
}


repeatability()
{
    : repeat the same cmd comparing log output 
    local py=${1:-2.7} 
    local cmd="$(ana_cmd $mode $py)" 
    echo $FUNCNAME : $py : $cmd  

    local log0="$(ana_log $mode $py 0)" 
    mkdir -p $(dirname $log0)

    local rc
    local nn=$(seq 0 9)
    for n in $nn ; do 
       local logn="$(ana_log $mode $py $n)"
       eval "$cmd" 2> $logn 1>&2 
       local diffcmd="diff $log0 $logn"
       eval $diffcmd
       rc=$?
       echo $diffcmd : $rc
       [ $rc -ne 0 ] && echo $FUNCNAME FAIL logn $logn && return 1 
    done
    return 0 
}


cfpy 
repeatability 2.7
repeatability 3

