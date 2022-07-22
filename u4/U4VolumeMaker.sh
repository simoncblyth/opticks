#!/bin/bash -l 

u4vdir=$(dirname $BASH_SOURCE)
defarg="grab"
arg=${1:-$defarg}


UBASE=/tmp/$USER/opticks/U4VolumeMaker_PVG_WriteNames_Sub

if [ "${arg/grab}" != "$arg" ]; then 
    source $u4vdir/../bin/rsync.sh $UBASE 
fi 





