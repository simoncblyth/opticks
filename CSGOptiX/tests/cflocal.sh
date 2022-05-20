#!/bin/bash -l 

export CFBASE_LOCAL=/tmp/$USER/opticks/GeoChain/BoxedSphere 

${IPYTHON:-ipython} --pdb -i cflocal.py 

