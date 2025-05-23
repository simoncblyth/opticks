#!/bin/bash 
usage(){ cat << EOU
pvplt_add_delta_lines_test.sh
==============================

Expected to create random point cloud with some red lines 

EOU
}

export MODE=3

script=pvplt_add_delta_lines_test.py
${IPYTHON:-ipython} --pdb -i $script


