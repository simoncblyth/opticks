#!/bin/bash -l 

MODE=${1:-0}

ana_cmd_0(){ cat << EOC
$* $(which tboolean.py) --tagoffset 0 --tag 1 --cat tboolean-box --pfx tboolean-box --src torch --show
EOC
}
ana_cmd_1(){ cat << EOC
LV=box $* $(which histype.py)
EOC
}
ana_cmd_2(){ cat << EOC
$* $(which seq.py)
EOC
}


usage(){ cat << EON
To regenerate the event files read by these analysis scripts, run::

    LV=box GGeoTest=INFO tboolean.sh --generateoverride 10000 --noviz

EON
}

ana_log(){ 
    local py="${*:-python3}"
    py=${py// /_}
    local mode=${MODE}
    local idx=${IDX:-0}
    echo /tmp/$USER/opticks/ana/ana_log_${py}_mode${mode}_idx${idx}.log
}

cfpy()
{
    : compare output from running same cmd with two pythons
    local py2cmd="$(ana_cmd_$MODE python2.7)"
    local py3cmd="$(ana_cmd_$MODE python3)"

    local py2log="$(ana_log python2.7)"
    local py3log="$(ana_log python3)"

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

run()
{
    local py="${*:-python3}" 
    echo py "$py"
    local cmd="$(ana_cmd_$MODE "$py")" 
    echo cmd "$cmd"
    local log0="$(ana_log "$py")" 
    echo log0 $log0
    mkdir -p $(dirname $log0)
    echo eval
    #eval "$cmd" 2> $log0 1>&2 
    eval "$cmd" 
    local rc=$?
    echo cmd $cmd : rc $rc 
    cat $log0
    echo cmd $cmd : rc $rc 
}


repeatability()
{
    : repeat the same cmd comparing log output 
    local py="${1:-python3}" 
    local cmd="$(ana_cmd_$MODE "$py")" 
    echo $FUNCNAME : "$py" : $cmd  

    local log0="$(ana_log "$py")" 
    mkdir -p $(dirname $log0)

    local rc
    local nn=$(seq 0 9)
    for n in $nn ; do 
       local logn="$(IDX=$n ana_log "$py")"
       eval "$cmd" 2> $logn 1>&2 
       local diffcmd="diff $log0 $logn"
       eval $diffcmd
       rc=$?
       echo $diffcmd : $rc
       [ $rc -ne 0 ] && echo $FUNCNAME FAIL logn $logn && return 1 
    done
    return 0 
}


#cfpy 
#repeatability python2.7
#repeatability python3

#run python2.7
#run python3
#run "ipython -i --"
run "/Users/blyth/miniconda3/bin/ipython -i --"
#run "/opt/local/bin/ipython -i --"
#run python 



