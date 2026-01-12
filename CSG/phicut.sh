#!/bin/bash

usage(){ cat << EOU

~/o/CSG/phicut.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=phicut
script=$name.py
${IPYTHON:-ipython} -i --pdb $script

