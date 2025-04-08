#!/bin/bash

upfind_cfbase(){
    : opticks/bin/upfind_cfbase.sh : traverse directory tree upwards searching the CFBase geometry dir identified by existance of relative CSGFoundry/solid.npy  
    local dir=$1
    while [ ${#dir} -gt 1 -a ! -f "$dir/CSGFoundry/solid.npy" ] ; do dir=$(dirname $dir) ; done 
    echo $dir
}

