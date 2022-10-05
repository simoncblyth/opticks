#!/bin/bash -l 

export GPropertyMap_BASE=/tmp/$USER/opticks/GEOM/ntds3/G4CXOpticks/GGeo/GScintillatorLib

name=GPropertyMap_make_table_Test

#$name

${IPYTHON:-ipython} --pdb -i $name.py 




