#!/bin/bash -l 


dir=/tmp/$USER/opticks/okop/OpSnapTest
mkdir -p $dir 

name=snap.jpg
cmd="scp P:$dir/$name $dir/$name"
echo $cmd
eval $cmd

path=$dir/$name
echo path $path 
open $path 




