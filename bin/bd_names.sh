#!/bin/bash 
usage(){ cat << EOU

~/o/bin/bd_names.sh 

EOU
}


SDIR=$(dirname $(realpath $BASH_SOURCE))

source $HOME/.opticks/GEOM/GEOM.sh 

cd $HOME/.opticks/GEOM/$GEOM/CSGFoundry/SSim/stree/standard

pwd

$SDIR/cat.py bd_names.txt

