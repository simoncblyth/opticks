#!/bin/bash -l 
usage(){ cat << EOU
AB_FOLD_COPY.sh 
================

srcbase : /tmp/$USER/opticks
dstbase : $OPTICKS_PREFIX/tests

If A_FOLD or B_FOLD is within srcbase this script 
will rsync the contents of the folders to dstbase 

::

   cd ~/opticks/bin
   ./AB_FOLD_COPY.sh 


EOU
}

source AB_FOLD.sh 

msg="=== $BASH_SOURCE :"
srcbase=/tmp/$USER/opticks
dstbase=$OPTICKS_PREFIX/tests

case $A_FOLD in 
  $srcbase*) A_REL=${A_FOLD/$srcbase\/} ; echo A_FOLD is within srcbase $srcbase A_REL $A_REL  ;; 
esac
if [ -n "$A_REL" ]; then 
    echo $msg sync to $dstbase/$A_REL
    mkdir -p $dstbase/$A_REL 
    rsync -av $srcbase/$A_REL/ $dstbase/$A_REL
    export A_FOLD_KEEP=$dstbase/$A_REL
fi 

case $B_FOLD in 
  $srcbase*) B_REL=${B_FOLD/$srcbase\/} ; echo B_FOLD is within srcbase $srcbase B_REL $B_REL  ;; 
esac  
if [ -n "$B_REL" ]; then 
    echo $msg sync to $dstbase/$B_REL
    mkdir -p $dstbase/$B_REL 
    rsync -av $srcbase/$B_REL/ $dstbase/$B_REL
    export B_FOLD_KEEP=$dstbase/$B_REL
fi 

echo $msg  A_FOLD_KEEP $A_FOLD_KEEP
echo $msg  B_FOLD_KEEP $B_FOLD_KEEP


