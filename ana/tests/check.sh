#!/bin/bash -l 

usage(){ cat << EOU
check.sh
==========

Setup environment:

PYTHONPATH=$HOME
   allows python scripts to import opticks python machinery 
   eg with  "from opticks.ana.fold import Fold"

CFBASE=$HOME/.opticks/GEOM/J004
   configures where to load geometry from

FOLD=$CFBASE/G4CXSimulateTest/ALL
   configures the directory to load event arrays from, 
   the directory is up to the user

To a large degree the directory positions of geometry 
and event files are controlled by the user. 
However the example of versioning a geometry name "J004"
and keeping event folders within the corresponding 
geometry folder is a good one to follow as it is important 
to retain the connection between event data and the geometry used
to create the event data.  

EOU
}

export PYTHONPATH=$HOME  
export CFBASE=$HOME/.opticks/GEOM/J004
export FOLD=$CFBASE/G4CXSimulateTest/ALL

${IPYTHON:-ipython} --pdb -i check.py 

