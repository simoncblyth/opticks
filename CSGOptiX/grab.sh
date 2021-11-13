#!/bin/bash -l 

usage(){ cat << EOU

./grab.sh 

REL=CSGOptiXSimulateTest/body_phys ./grab.sh 



EOU
}

name=$(basename $(pwd))

if [ -z "$REL" ]; then
   pkg=$name
else
   pkg=$name/$REL
fi 

msg="=== $BASH_SOURCE :"


from=P:/tmp/$USER/opticks/$pkg/
to=/tmp/$USER/opticks/$pkg/


echo $msg name $name REL $REL pkg $pkg from $from to $to


mkdir -p $to

if [ "$1" != "ls" ]; then
rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
fi 

ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `

