#!/bin/bash -l 

dir=$(dirname $BASH_SOURCE)
path=$dir/tests/X4MeshTest.py

echo BASH_SOURCE $BASH_SOURCE path $path 

${IPYTHON:-ipython} --pdb $path


