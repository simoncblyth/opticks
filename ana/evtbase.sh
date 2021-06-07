#!/bin/bash -l 

CMD=$1

usage(){ cat << EOU
evtbase.sh 
===========

rsyncs persisted OpticksEvent between machines 

EOU
}



REMOTE=${REMOTE:-P}
OPTICKS_EVENT_BASE_REMOTE=${OPTICKS_EVENT_BASE_REMOTE:-/home/$USER/local/opticks/evtbase}
OPTICKS_EVENT_BASE=${OPTICKS_EVENT_BASE:-/tmp/$USER/opticks}

reldirs_(){ cat << EOR
source/evt/g4live/natural
tds3ip/evt/g4live/natural
EOR
}

sync()
{
   local msg="=== $FUNCNAME :"
   local reldir=${1}
   [ -z "$reldir" ] && echo $msg ERROR missing reldir arg && return 1 

   local from=$REMOTE:${OPTICKS_EVENT_BASE_REMOTE}/$reldir
   local to=${OPTICKS_EVENT_BASE}/$reldir

    echo $FUNCNAME from $from to $to

    if [ "$CMD" == "rm" -a -d "$to" ]; then 
       rm -rf $to   
    fi
    mkdir -p $to 

    if [ "$CMD" != "ls" ]; then
        rsync -zarv --progress --include="*/" --include="*.npy" --include="*.txt" --include="*.json" --include="*.ini" --exclude="*" "${from}/" "${to}/"
    fi 

    ls -1rt `find ${to%/} -name '*.npy' `

    return 0   
}


for reldir in $(reldirs_) ; do echo $reldir ; done 
for reldir in $(reldirs_) ; do sync $reldir ; done 



