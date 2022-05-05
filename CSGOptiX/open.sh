#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

defpath=/tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/SCVD1/70000/-1/SGLM_-1.jpg
#defpath=/tmp/blyth/opticks/CSGOptiX/CSGOptiXRenderTest/SCVD1/70000/-1/Comp_-1.jpg

path=${1:-$defpath}

echo $msg path $path 

if [ ! -f "$path" ]; then 
   echo $msg scp from remote 
   mkdir -p $(dirname $path)
   scp P:$path $path
fi 

open $path 
