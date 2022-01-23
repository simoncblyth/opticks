#!/bin/bash -l 

geom="UnionBoxSphere"

export GEOM=${GEOM:-$geom}

name=CSGSignedDistanceFieldTest
which $name

$name
[ $? -ne 0 ] && echo $msg runtime error && exit 1 


${IPYTHON:-ipython} --pdb -i $name.py 
[ $? -ne 0 ] && echo $msg ana error && exit 2

exit 0  


