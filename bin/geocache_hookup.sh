#!/bin/bash -l 
usage(){ cat << EON
geocache_hookup.sh 
====================

The geocache_hookup argument is used to select the geometry:

old
    some old reference geometry 
new
    recent addition
last
    last created geocache, using the OPTICKS_KEY export line 
    written by Opticks::writeGeocacheScript writes::

        /usr/local/opticks/geocache/geocache.sh 
        ~/.opticks/geocache/geocache.sh

EON
}

msg="=== $BASH_SOURCE :"

geocache_ls()
{
    echo $msg OPTICKS_KEY $OPTICKS_KEY 
    local cfbase=${OPTICKS_KEYDIR}/CSG_GGeo
    local logdir=$cfbase/logs  # matches the chdir done in tests/CSG_GGeoTest.cc
    local outdir=$cfbase/CSGFoundry

    echo $msg outdir:$outdir
    ls -l $outdir/

    echo $msg logdir:$logdir
    ls -l $logdir/
}

last_geocache_sh(){
    local opticks_geocache_prefix=$HOME/.opticks
    local geocache_sh=${OPTICKS_GEOCACHE_PREFIX:-$opticks_geocache_prefix}/geocache/geocache.sh
    echo $geocache_sh
}

geocache_hookup_last()
{
    local geocache_sh=$(last_geocache_sh)
    if [ -f "$geocache_sh" ]; then
        echo $msg sourcing geocache_sh $geocache_sh that was written by Opticks::writeGeocacheScript
        ls -alst $geocache_sh
        #cat $geocache_sh
        source $geocache_sh

        export OPTICKS_KEYFUNC=$FUNCNAME
    else
        echo $msg ERROR expecting to find geocache_sh $geocache_sh
    fi 
}

geocache_hookup()
{
    local arg=${1:-new}
    export OPTICKS_GEOCACHE_HOOKUP_ARG=$arg
    case $arg in  
       old)  keyfunc=geocache-12dec2021-key ;;
       new)  keyfunc=geocache-02mar2022-key ;;
      last)  keyfunc=last ;;
         *)  keyfunc=geocache-02mar2022-key ;;
    esac

    if [ "$keyfunc" == "last" ]; then 
        geocache_hookup_last 
    else 
        export OPTICKS_KEYFUNC=$keyfunc
        export OPTICKS_KEY=$(geocache-;${OPTICKS_KEYFUNC})
    fi 

    local vars="arg keyfunc OPTICKS_KEYFUNC OPTICKS_KEY"
    for var in $vars ; do printf "%20s : %s \n" $var ${!var} ; done
}

geocache_hookup $*


