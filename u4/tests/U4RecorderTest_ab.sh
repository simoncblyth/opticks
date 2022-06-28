#!/bin/bash -l 

source ../../bin/AB_FOLD.sh 


${IPYTHON:-ipython} --pdb -i U4RecorderTest_ab.py $*  




