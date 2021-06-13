#!/bin/bash -l 

name=$(basename $BASH_SOURCE)
pfx=${name/.sh}

arg=${1:-1}
shift 
args=$* 


if [ "$arg" == "sync" -o "$arg" == "get" ]; then
    cmd="PFX=$pfx evtsync.sh" 
else
    tag=$arg 
    cmd="PFX=$pfx ab.sh $tag $args"
fi

echo $cmd
eval $cmd 

