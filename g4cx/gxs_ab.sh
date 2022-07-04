#!/bin/bash -l 
usage(){ cat << EOU
gxs_ab.sh
=======================

::

    gx
    ./gxs_ab.sh

EOU
}


#fold_mode=TMP
#fold_mode=KEEP
#fold_mode=LOGF
#fold_mode=GEOM
fold_mode=GXS

export FOLD_MODE=${FOLD_MODE:-$fold_mode}

source ../bin/AB_FOLD.sh 

${IPYTHON:-ipython} --pdb -i tests/G4CXSimulateTest_ab.py $*  



