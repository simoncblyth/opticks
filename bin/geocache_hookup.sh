#!/bin/bash -l 
usage(){ cat << EON
geocache_hookup.sh 
====================

Opticks::writeGeocacheScript writes::

    /usr/local/opticks/geocache/geocache.sh 
    ~/.opticks/geocache/geocache.sh

The script contains the "export OPTICKS_KEY=.." 
of the geocache just created.
EON
}

msg="=== $BASH_SOURCE :"

geocache_ls()
{
    local cfbase=${OPTICKS_KEYDIR}/CSG_GGeo
    local logdir=$cfbase/logs  # matches the chdir done in tests/CSG_GGeoTest.cc
    local outdir=$cfbase/CSGFoundry

    echo $msg outdir:$outdir
    ls -l $outdir/

    echo $msg logdir:$logdir
    ls -l $logdir/
}

opticks_geocache_prefix=$HOME/.opticks
geocache_sh=${OPTICKS_GEOCACHE_PREFIX:-$opticks_geocache_prefix}/geocache/geocache.sh

if [ -f "$geocache_sh" ]; then
    echo $msg sourcing geocache_sh $geocache_sh that was written by Opticks::writeGeocacheScript
    ls -alst $geocache_sh
    cat $geocache_sh
    source $geocache_sh
    echo $msg OPTICKS_KEY $OPTICKS_KEY 
    geocache_ls 
else
    echo $msg ERROR expecting to find geocache_sh $geocache_sh
fi 


