#!/bin/bash -l 

source ../bin/AB_FOLD.sh 

${IPYTHON:-ipython} --pdb -i dv.py $*  


