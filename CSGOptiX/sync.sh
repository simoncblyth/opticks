#!/bin/bash -l 

name=$(basename $PWD)
cmd="rsync -rtz --progress --exclude='__pycache__/' --exclude='.git/' --exclude='*.swp' --exclude='*.pyc'  $PWD/ P:$name/"
echo $cmd
eval $cmd
