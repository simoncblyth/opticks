#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
export GPropertyMap_BASE=/tmp/$USER/opticks/GEOM/ntds3/G4CXOpticks/GGeo/GScintillatorLib
name=GPropertyMap_make_table_Test

defarg=run
arg=${1:-$defarg}

if [ "${arg/run}" != "$arg" ]; then 
   $name
   [ $? -ne 0 ] && echo $msg run fail && exit 1
fi

if [ "${arg/ana}" != "$arg" ]; then 
   ${IPYTHON:-ipython} --pdb -i $name.py 
   [ $? -ne 0 ] && echo $msg ana fail && exit 2
fi 




