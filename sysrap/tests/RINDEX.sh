#!/bin/bash -l 
SDIR=$(cd $(dirname $BASH_SOURCE) && pwd)

name=RINDEX
source $HOME/.opticks/GEOM/GEOM.sh 
export material=LS

${IPYTHON:-ipython} -i --pdb $SDIR/$name.py 

