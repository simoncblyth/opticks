#!/bin/bash -l 
usage(){ cat << EOU
pull.sh
==========

When pull times gets tediously slow, 
can instead use::

   ~/opticks/bin/rsync_put.sh 

EOU
}

git remote -v

date 
git pull
date 


