#!/bin/bash -l 

default=remote
arg=${1:-$default}

if [ "$arg" == "remote" ]; then 
   export OPTICKS_KEY=$OPTICKS_KEY_REMOTE 
   export OPTICKS_GEOCACHE_PREFIX=$HOME/.opticks 
elif [ "$arg" == "old" ]; then 
   export OPTICKS_KEY=$OPTICKS_KEY_OLD
elif [ "$arg" == "new" ]; then 
   export OPTICKS_KEY=$OPTICKS_KEY_NEW
elif [ "$arg" == "asis" ]; then 
   echo $msg using OPTICKS_KEY : $OPTICKS_KEY 
fi 

ipython -i CSGFoundry.py 

