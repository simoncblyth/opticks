#!/bin/bash -l 

EVT="%0.3d"

export GEOM=V0J008
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/ntds2
export FOLD=$BASE/ALL0/$EVT

echo FOLD $FOLD

${IPYTHON:-ipython} --pdb -i fold.py 
