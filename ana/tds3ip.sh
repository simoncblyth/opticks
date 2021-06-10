#!/bin/bash -l 

tag=${1:-1}
shift 
args=$* 

name=$(basename $BASH_SOURCE)
pfx=${name/.sh}
echo $name

cmd="PFX=$pfx ab.sh $tag $args"

echo $cmd
eval $cmd 
