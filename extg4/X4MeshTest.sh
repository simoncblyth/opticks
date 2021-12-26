#!/bin/bash -l 

usage(){ cat << EOU

GEOM=hmsk_solidMask     EYE=0.5,0.5,-0.3 ZOOM=2 ./X4MeshTest.sh
GEOM=hmsk_solidMaskTail EYE=0.5,0.5,0.3 ZOOM=2 ./X4MeshTest.sh


GEOM=nmsk_solidMaskTail EYE=0.5,0.5,0.3 ZOOM=2 ./X4MeshTest.sh


EOU
}


msg="=== $BASH_SOURCE :"

[ -z "$GEOM" ] && echo $msg Mandatory GEOM envvar is missing && exit 1 


dir=$(dirname $BASH_SOURCE)
bin=$(which X4MeshTest)
script=$dir/tests/X4MeshTest.py

geom=${GEOM}
outdir="$TMP/extg4/X4MeshTest/$geom/X4Mesh"

echo BASH_SOURCE $BASH_SOURCE bin $bin script $script outdir $outdir

$bin

ls -l $outdir 


${IPYTHON:-ipython} --pdb -i $script


exit 1

