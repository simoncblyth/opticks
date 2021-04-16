#!/bin/bash -l 

dir=$(dirname $BASH_SOURCE)
name=$(basename $BASH_SOURCE)
stem=${name/.sh}

cmd="${OPTICKS_PYTHON:-python3} $dir/$stem.py $*"

echo $cmd
eval $cmd 

