#!/bin/bash -l 

reldir=source/evt/g4live/natural
RELDIR=${RELDIR:-$reldir}

REMOTE=${REMOTE:-P}
OPTICKS_EVENT_BASE_REMOTE=${OPTICKS_EVENT_BASE_REMOTE:-/home/$USER/local/opticks/evtbase}
OPTICKS_EVENT_BASE=${OPTICKS_EVENT_BASE:-/tmp/$USER/opticks}

from=$REMOTE:${OPTICKS_EVENT_BASE_REMOTE}/$RELDIR
to=${OPTICKS_EVENT_BASE}/$RELDIR

echo from $from to $to

if [ "$1" == "rm" ]; then 
   rm -rf $to   
fi
mkdir -p $to 

if [ "$1" != "ls" ]; then
rsync -zarv --progress --include="*/" --include="*.npy" --include="*.txt" --include="*.json" --include="*.ini" --exclude="*" "${from}/" "${to}/"
fi 

ls -1rt `find ${to%/} -name '*.npy' `

