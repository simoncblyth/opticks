#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
hhs="NP.hh NPU.hh"
for hh in $hhs ; do 
    cmd="diff $HOME/np/$hh $hh"
    echo $msg $cmd
    eval $cmd
    [ $? -ne 0 ] && echo $msg DIFF DETECTED && exit 1  
done

exit 0 

