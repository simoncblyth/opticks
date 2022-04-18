#!/bin/bash -l 

arg=${1:-all}
shift

msg="=== $BASH_SOURCE :"
executable=GeoChain
EXECUTABLE=${EXECUTABLE:-$executable}
xdir=/tmp/blyth/opticks/$EXECUTABLE/     ## NB trailing slash to avoid rsync duplicating path element 

from=P:$xdir
to=$xdir

LOGDIR=/tmp/$USER/opticks/CSGOptiX/CSGOptiXSimtraceTest


printf "arg                    %s  ( the  possibilities are : png jpg mp4 all ) \n" "$arg"
printf "EXECUTABLE             %s \n " "$EXECUTABLE"
printf "\n"
printf "xdir                   %s \n" "$xdir"
printf "from                   %s \n" "$from" 
printf "to                     %s \n" "$to" 

mkdir -p $to


grab_typ(){
   local typ=${1:-png}

    rsync -zarv --progress --include="*/" --include="*.$typ" --include="*.json" --exclude="*" "$from" "$to"
    local typfind=$(find ${to%/} -name "*.$typ")

    if [ -n "$typfind" ]; then
        ls -1rt $typfind
        local last=$(ls -1rt $typfind  | tail -1 )
        echo $msg typ $typ last $last
        if [ "$(uname)" == "Darwin" ]; then
            open $last 
        fi
    else
        echo $msg failed to find $typ
    fi 
}

grab_all()
{
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `

    local all_npy=$(find ${to%/} -name '*.npy')
    
    if [ -n "$all_npy" ]; then 

        local last_npy=$(ls -1rt $all_npy | tail -1 )
        local last_outdir=$(dirname $last_npy)

        if [ ! -d "$LOGDIR" ]; then 
            echo $msg creating LOGDIR $LOGDIR
            mkdir -p $LOGDIR 
        fi 

        if [ -d "$LOGDIR" ]; then 
            local script=$LOGDIR/CSGOptiXSimtraceTest_OUTPUT_DIR.sh
            printf "export CSGOptiXSimtraceTest_OUTPUT_DIR=$last_outdir\n" > $script 
            echo $msg script $script
            cat $script
        else
            echo $msg LOGDIR $LOGDIR does not exist : cannot write CSGOptiXSimtraceTest_OUTPUT_DIR.sh based on last_outdir $last_outdir : use arg like jpg to grab renders only 
        fi 

    fi

}


echo 
read -p "$msg Enter YES to proceed with rsync grab from remote : " ans
if [ "$ans" == "YES" ]; then 
    echo $msg PROCEEDING  
else
    echo $msg SKIPPING : perhaps you should use cxs_grab.sh or another script ?
    exit 1 
fi 


case $arg in 
   png) grab_typ png ;; 
   jpg) grab_typ jpg ;; 
   mp4) grab_typ mp4 ;; 
   all) grab_all     ;;
esac


