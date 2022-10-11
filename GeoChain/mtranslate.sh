#!/bin/bash -l 
usage(){ cat << EOU
mtranslate.sh 
=====================

Workflow:

1. change the GEOM list with *geomlist_* bash function
2. check GEOM names with dry run::

    ~/opticks/GeoChain/mtranslate.sh dry

3. do the translations::

    ~/opticks/GeoChain/mtranslate.sh

::

    epsilon:~ blyth$ l /tmp/blyth/opticks/GEOM/
    total 0
    0 drwxr-xr-x   3 blyth  wheel   96 Oct 11 11:32 hamaInner2Solid__U1
    0 drwxr-xr-x  14 blyth  wheel  448 Oct 11 11:32 .
    0 drwxr-xr-x   3 blyth  wheel   96 Oct 11 11:32 hamaInner1Solid__U1
    0 drwxr-xr-x   3 blyth  wheel   96 Oct 11 11:32 hamaBodySolid__U1
    0 drwxr-xr-x   3 blyth  wheel   96 Oct 11 11:32 hamaPMTSolid__U1
    0 drwxr-xr-x   4 blyth  wheel  128 Oct 11 11:32 hmskSolidMaskTail__U1
    0 drwxr-xr-x   4 blyth  wheel  128 Oct 11 11:32 hmskSolidMask__U1
    0 drwxr-xr-x   4 blyth  wheel  128 Oct 11 11:32 hmskSolidMaskVirtual__U1
    0 drwxr-xr-x   5 blyth  wheel  160 Oct 11 10:50 nmskSolidMaskVirtual__U1
    0 drwxr-xr-x   5 blyth  wheel  160 Oct 11 10:25 nmskSolidMaskTail__U1
    0 drwxr-xr-x   4 blyth  wheel  128 Oct 11 10:08 nnvtPMTSolid__U1
    0 drwxr-xr-x   4 blyth  wheel  128 Oct 11 09:30 ..
    0 drwxr-xr-x   3 blyth  wheel   96 Oct 10 20:45 nmskSolidMask__U1
    0 drwxr-xr-x   3 blyth  wheel   96 Oct  5 11:28 ntds3
    epsilon:~ blyth$ 


EOU
}

geomlist_OPT=U1
names=$(source $(dirname $BASH_SOURCE)/../bin/geomlist.sh names) 

defarg="run"
arg=${1:-$defarg}

if [ "$arg" == "dry" ]; then
    for geom in $names ; do 
       echo $BASH_SOURCE geom $geom opt $opt 
    done 
fi 

if [ "$arg" == "run" ]; then
    for geom in $names ; do 
       echo $BASH_SOURCE geom $geom opt $opt 
       GEOM=${geom} $(dirname $BASH_SOURCE)/../GeoChain/translate.sh 
       [ $? -ne 0 ] && echo $BASH_SOURCE translate error for geom $geom && exit 1
    done 
fi

exit 0


