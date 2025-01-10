#!/bin/bash 

usage(){ cat << EOU

~/o/sysrap/tests/sgenstep__test.sh 

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

name=sgenstep__test
script=$name.py

defarg="info_pdb"
arg=${1:-$defarg}


source $HOME/.opticks/GEOM/GEOM.sh 
source $HOME/.opticks/CTX/CTX.sh 
source $HOME/.opticks/TEST/TEST.sh 

RELDIR=${CTX}_${TEST}

export GSFOLD=$TMP/GEOM/$GEOM/jok-tds/$RELDIR/A000_OIM1
export GSPATH=$GSFOLD/genstep.npy

vars="BASH_SOURCE defarg arg name script GEOM CTX TEST RELDIR GSFOLD GSPATH"

if [ "${arg/info}" != "$arg" ]; then
   for var in $vars ; do printf "%20s : %s\n" "$var" "${!var}" ; done 
fi 

if [ "${arg/pdb}" != "$arg" ]; then
   ${IPYTHON:-ipython} --pdb -i $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE pdb error && exit 1 
fi 

if [ "${arg/ana}" != "$arg" ]; then
   ${PYTHON:-python} $script 
   [ $? -ne 0 ] && echo $BASH_SOURCE ana error && exit 2 
fi 

if [ "${arg/grab}" != "$arg" ]; then 
    source ../../bin/rsync.sh $GSFOLD  
    [ $? -ne 0 ] && echo $BASH_SOURCE : grab error && exit 3
fi

exit 0 

