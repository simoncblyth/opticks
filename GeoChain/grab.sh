#!/bin/bash -l 

usage(){ cat << EOU

When envvar GEOM is defined the grab is restricted to just that subdir 
making it a bit faster.

EOU
}

name=$(basename $(pwd))
if [ -z "$GEOM" ]; then
    rel=$name
else
    rel=$name/$GEOM
fi 

from=P:/tmp/$USER/opticks/$rel/
to=/tmp/$USER/opticks/$rel/

mkdir -p $to

msg="=== $BASH_SOURCE :"
echo $msg name $name GEOM $GEOM rel $rel from $from to $to

if [ "$1" != "ls" ]; then
rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
fi 


echo $msh ls0 : json txt
ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `

echo $msh ls1 : jpg mp4 npy
ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `

