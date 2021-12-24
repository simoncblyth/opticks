#!/bin/bash -l 

usage(){ cat << EOU

GEOM=hmsk_solidMaskTail CPOS=0.5,0.5,0.3 ./X4MeshTest.sh

EOU
}


dir=$(dirname $BASH_SOURCE)
path=$dir/tests/X4MeshTest.py

echo BASH_SOURCE $BASH_SOURCE path $path 

${IPYTHON:-ipython} --pdb -i $path


