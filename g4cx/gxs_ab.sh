#!/bin/bash -l 
usage(){ cat << EOU
gxs_ab.sh
=======================

::

    gx
    ./gxs_ab.sh

EOU
}

source ../bin/GEOM_.sh 

FOLD_MODE=GXS source ../bin/AB_FOLD.sh 

${IPYTHON:-ipython} --pdb -i tests/G4CXSimulateTest_ab.py $*  


