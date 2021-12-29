#!/bin/bash -l 

usage(){ cat << EOU

GEOM=hmsk_solidMask     EYE=0.5,0.5,-0.3 ZOOM=2 ./X4MeshTest.sh
GEOM=hmsk_solidMaskTail EYE=0.5,0.5,0.3 ZOOM=2 ./X4MeshTest.sh


GEOM=nmsk_solidMaskTail EYE=0.5,0.5,0.3 ZOOM=2 ./X4MeshTest.sh


GEOM=XJfixtureConstruction ./X4MeshTest.sh
GEOM=XJanchorConstruction ./X4MeshTest.sh


EOU
}


msg="=== $BASH_SOURCE :"


dir=$(dirname $BASH_SOURCE)
bin=$(which X4MeshTest)
script=$dir/tests/X4MeshTest.py


geom=XJfixtureConstruction 
export GEOM=${GEOM:-$geom}

outdir="$TMP/extg4/X4MeshTest/$GEOM/X4Mesh"

if [ "$GEOM" == "XJfixtureConstruction" ]; then
    source XJfixtureConstruction.sh
fi 

echo BASH_SOURCE $BASH_SOURCE bin $bin script $script outdir $outdir GEOM $GEOM

$bin

ls -l $outdir 


${IPYTHON:-ipython} --pdb -i $script


exit 1

