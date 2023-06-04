#!/bin/bash -l 

arg=${1:-0}

DIR=$(dirname $BASH_SOURCE)
GEOMDIR=$HOME/.opticks/GEOM

source $GEOMDIR/GEOM.sh 

[ -z "$GEOM" ] && echo $BASH_SOURCE : ERROR NO GEOM && exit 1 

path=$HOME/.opticks/GEOM/$GEOM/CSGFoundry/meshname.txt

[ ! -f "$path" ] && echo $BASH_SOURCE : ERROT NO path $path && exit 2 

echo $path 
$DIR/cat.py -s $arg $path

exit 0 

