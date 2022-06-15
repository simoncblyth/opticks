#!/bin/bash -l 

source ./IDPath_override.sh 

export MATERIAL=Air

U4MaterialPropertyVectorTest 

export FOLD=/tmp/$USER/opticks/U4MaterialPropertyVectorTest

${IPYTHON:-ipython} --pdb -i U4MaterialPropertyVectorTest.py 





