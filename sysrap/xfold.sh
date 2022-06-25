#!/bin/bash -l 

export A_FOLD=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
export B_FOLD=/tmp/$USER/opticks/U4RecorderTest
export FOLD=$A_FOLD

${IPYTHON:-ipython} --pdb -i xfold.py 
