#!/bin/bash -l 

usage(){ cat << EOU
U4RecorderTest_ab.sh
=======================

::

    u4t
    ./U4RecorderTest_ab.sh


EOU
}

source ../../bin/AB_FOLD.sh 

${IPYTHON:-ipython} --pdb -i U4RecorderTest_ab.py $*  




