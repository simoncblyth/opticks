#!/bin/bash 

tmp=/tmp/$USER/opticks
export TMP=${TMP:-$tmp}
mkdir -p $TMP

../ana/enum_.py OpticksGenstep.h --quiet --simple --inipath $TMP/OpticksGenstep_Enum.ini && cat $TMP/OpticksGenstep_Enum.ini
[ $? -ne 0 ] && echo $BASH_SOURCE ERROR parsing OpticksGenstep.h && exit 1

../ana/enum_.py OpticksPhoton.h --quiet --inipath $TMP/OpticksPhoton_Enum.ini && cat $TMP/OpticksPhoton_Enum.ini
[ $? -ne 0 ] && echo $BASH_SOURCE ERROR parsing OpticksPhoton.h && exit 2

exit 0


