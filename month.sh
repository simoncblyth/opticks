#!/bin/bash -l 
usage(){ cat << EOU
mo.sh : git commit messages by month or year for this year and last
=====================================================================

::

   o
   ./mo.sh -h   # this help message    

   ./mo.sh 0    # all this year 
   ./mo.sh -0   # all last year 

   ./mo.sh  1    # this year january 
   ./mo.sh -1    # last year january  
   ./mo.sh -2    # last year february 
   ./mo.sh -12   # last year december

EOU
}


mo=${1:-1}
rev=${REV:-0}

year=$(date +"%Y")
lastyear=$(( $year - 1 ))

if [ "${mo:0:1}" == "-" ]; then
    mo=${mo:1}
    logyear=$lastyear 
else
    logyear=$year 
fi 

YEAR=${YEAR:-$logyear}


[ "$mo" == "h" ] && usage && exit 0 

log=lg
LOG=${LOG:-$log}

month=$(printf "%0.2d" $mo)

if [ "$mo" == "0" ]; then 
    cmd="git $LOG --after $YEAR-01-01 --before $YEAR-12-31"

else 
    cmd="git $LOG --after $YEAR-$month-01 --before $YEAR-$month-31"
fi 

if [ "$rev" == "1" ]; then 
   cmd="$cmd | tail -r"
fi 


echo $cmd
eval $cmd


exit 0 
