#!/bin/bash -l 

usage(){ cat << EOU
stats.sh
==========

::

    cd ~/opticks/bin
    ./stats.sh 



EOU
}

#${IPYTHON:-ipython} --pdb -i stats.py 

${IPYTHON:-ipython} --pdb -i packages.py 




