#!/bin/bash
usage(){ cat << EOU
pvplt_add_delta_lines_test.sh
==============================

Expected to create random point cloud with some red lines

EOU
}

export MODE=3

#size=1024,768
#size=1280,720   ## default
#size=1280,720   ## default
size=2560,1440  ##

export SIZE=${SIZE:-$size}   ## it gets doubled

script=pvplt_add_delta_lines_test.py
${IPYTHON:-ipython} --pdb -i $script


