#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
arg=${1:-diff}
echo $msg arg [$arg] AVAILABLE CMDS : diff, cp 
hhs="NP.hh NPU.hh"

if [ "$arg" == "diff" ]; then 
    for hh in $hhs ; do 
        cmd="diff $HOME/np/$hh $hh"
        echo $msg $cmd
        eval $cmd
        [ $? -ne 0 ] && echo $msg DIFF DETECTED && exit 1  
    done
elif [ "$arg" == "cp" ]; then 
    for hh in $hhs ; do 
        cmd="cp $HOME/np/$hh $hh"
        echo $msg $cmd
        eval $cmd
        [ $? -ne 0 ] && echo $msg COPY ERROR && exit 2  
    done
fi 


exit 0 

