#!/bin/bash -l 

usage(){ cat << EOU
U4RecorderTest_ab.sh
=======================

::

    u4t
    ./U4RecorderTest_ab.sh


EOU
}


#fold_mode=TMP
#fold_mode=KEEP
fold_mode=LOGF

export FOLD_MODE=${FOLD_MODE:-$fold_mode}

source ../../bin/AB_FOLD.sh 

${IPYTHON:-ipython} --pdb -i U4RecorderTest_ab.py $*  




