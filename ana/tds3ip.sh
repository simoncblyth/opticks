#!/bin/bash -l 

usage(){ cat << EOU

The prefix PFX=tds3gun is obtained from the name of this script 

tds3gun.sh get 
    grabs remote events to local with: ``PFX=tds3gun evtsync.sh`` 

tds3gun.sh 1 
    runs ``PFX=tds3gun.sh ab.sh 1`` comparing events with tags 1 and -1

EOU
}



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

