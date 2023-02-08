#!/bin/bash -l 
usage(){ cat << EOU
findpath.sh
============

Finds last modified paths with extensions.
Add extension scripts by creating symbolic links eg::

    cd ~/opticks/bin

    ln -s findpath.sh rst.sh 
    ln -s findpath.sh txt.sh 
    ln -s findpath.sh cc.sh 
 

EOU
}

name=$(basename $BASH_SOURCE)
name=${name/.sh}
ext=$name

SDIR=$(dirname $BASH_SOURCE)

last=20

export LAST=${LAST:-$last}
export EXT=${EXT:-$ext}


#IOPT=-i

${IPYTHON:-ipython} --pdb $IOPT $(dirname $BASH_SOURCE)/findpath.py 


