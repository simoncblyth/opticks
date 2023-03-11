#!/bin/bash -l 

name=ModelTrigger_Debug

export GEOM=FewPMT
export BASE=/tmp/$USER/opticks/GEOM/$GEOM/U4SimulateTest
export VERSION=${N:-0}
export MODE=2

reldir=ALL$VERSION

export FOLD=$BASE/$reldir
${IPYTHON:-ipython} --pdb -i ${name}.py 
[ $? -ne 0 ] && echo $BASH_SOURCE $name error && exit 1


exit 0 


