#!/bin/bash

usage(){ cat << EOU

~/o/ana/contrib_bb.sh

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=contrib_bb
script=$name.py

${IPYTHON:-ipython} -i --pdb $script
