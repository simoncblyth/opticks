#!/bin/bash -l 

usage(){ cat << EOU
mtranslate.sh 
=====================

::

    gc
    vi mtranslate.sh    # change the list of GEOM to translate using GeoChain test machinery 
    ./mtranslate.sh 

EOU
}

geomlist_OPT=U1
names=$(source $(dirname $BASH_SOURCE)/../bin/geomlist.sh names)  # use geomlist to edit the geomlist bash functions 

for geom in $names ; do 
   echo $BASH_SOURCE geom $geom opt $opt 
   GEOM=${geom} ./translate.sh 
   [ $? -ne 0 ] && echo $BASH_SOURCE translate error for geom $geom && exit 1
done 

exit 0


