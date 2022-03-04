#!/bin/bash -l 

msg="=== $BASH_SOURCE :"

geocache_sh=${OPTICKS_GEOCACHE_PREFIX:-$HOME/.opticks}/geocache/geocache.sh

if [ -f "$geocache_sh" ]; then
    echo $msg sourcing geocache_sh $geocache_sh that was written by Opticks::writeGeocacheScript
    ls -alst $geocache_sh
    source $geocache_sh
    echo $msg OPTICKS_KEY $OPTICKS_KEY 
else
    echo $msg ERROR expecting to find geocache_sh $geocache_sh
fi 


