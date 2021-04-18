#!/bin/bash -l 



dir=/tmp/$USER/opticks/okop/OpSnapTest
mkdir -p $dir 

if [ "$1" == "grab" ]; then 
    cmd="rsync -rtz --del --progress P:$dir/ $dir/"
    echo $cmd
    eval $cmd
fi 

open $dir




