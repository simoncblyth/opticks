#!/bin/bash 

usage(){ cat << EOU

~/o/sysrap/tests/sgenstep__test.sh 
~/o/sysrap/tests/sgenstep__test.sh pvcap
PUB=mu214gev ~/o/sysrap/tests/sgenstep__test.sh pvpub

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

vars="BASH_SOURCE defarg arg name script GEOM CTX TEST RELDIR GSFOLD GSPATH MODE"

case $(uname) in
  Darwin) mode=3 ;;
  Linux)  mode=0 ;;
esac
export MODE=${MODE:-$mode}
export NOGRID=1

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

if [ "$arg" == "pvcap" -o "$arg" == "pvpub" -o "$arg" == "mpcap" -o "$arg" == "mppub" ]; then
    export CAP_BASE=$GSFOLD/figs
    export CAP_REL=sgenstep__test
    export CAP_STEM=$RELDIR
    case $arg in  
       pvcap) source pvcap.sh cap  ;;  
       mpcap) source mpcap.sh cap  ;;  
       pvpub) source pvcap.sh env  ;;  
       mppub) source mpcap.sh env  ;;  
    esac
    if [ "$arg" == "pvpub" -o "$arg" == "mppub" ]; then 
        source epub.sh 
    fi  
fi 







exit 0 

