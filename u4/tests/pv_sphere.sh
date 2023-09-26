#!/bin/bash -l 

cd $(dirname $BASH_SOURCE) 
name=pv_sphere
${IPYTHON:-ipython} --pdb -i $name.py 
