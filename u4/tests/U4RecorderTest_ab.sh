#!/bin/bash -l 

export A_FOLD=/tmp/$USER/opticks/GeoChain/BoxedSphere/CXRaindropTest
export B_FOLD=/tmp/$USER/opticks/U4RecorderTest

${IPYTHON:-ipython} --pdb -i U4RecorderTest_ab.py $*  




