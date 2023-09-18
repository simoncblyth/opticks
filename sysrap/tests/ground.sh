#!/bin/bash -l 

GEOM top 
pwd

for m in $(grep -l ground CSGFoundry/SSim/stree/surface/*/NPFold_meta.txt)
do 
    echo 
    echo $m 
    cat $m 
    dir=$(dirname $m)
    ls -l $dir/*.npy
done 




