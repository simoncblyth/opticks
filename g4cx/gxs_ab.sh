#!/bin/bash -l 
usage(){ cat << EOU
gxs_ab.sh
=======================

::

    gx
    ./gxs_ab.sh

EOU
}


source $(dirname $BASH_SOURCE)/../bin/COMMON.sh 

FOLD_MODE=GXS source $(dirname $BASH_SOURCE)/../bin/AB_FOLD.sh 

export CFBASE 

${IPYTHON:-ipython} --pdb -i tests/G4CXSimulateTest_ab.py $*  


