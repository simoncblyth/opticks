#!/bin/bash -l 

path=${BASH_SOURCE}
name=$(basename $path)
stem=${name/.sh}
echo path $path name $name stem $stem

fold=/tmp/QPropTest
mkdir -p $fold/float
mkdir -p $fold/double

which $stem
$stem

ipython -i $stem.py



