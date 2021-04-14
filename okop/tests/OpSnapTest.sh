#!/bin/bash -l 


dir=/tmp/$USER/opticks/okop/OpSnapTest
mkdir -p $dir 

cmd="rsync -rtz --del --progress P:$dir/ $dir/"
echo $cmd
eval $cmd

open $dir




