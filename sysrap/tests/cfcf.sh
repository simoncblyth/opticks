#!/bin/bash -l 

export STBASE=/tmp/$USER/opticks/ntds3/G4CXOpticks
export STBASE_aug5=/tmp/$USER/opticks/ntds3_aug5/G4CXOpticks


export A_CFBASE=$STBASE_aug5
export B_CFBASE=$STBASE

arg=ana


if [ "${arg/ana}" != "$arg" ]; then 
    ${IPYTHON:-ipython} --pdb -i cfcf.py 
    [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

exit 0 

