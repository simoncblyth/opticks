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

[ "$mo" == "h" ] && usage && exit 0 


month=$(printf "%0.2d" $mo)

if [ "$mo" == "0" ]; then 
    cmd="git lg --after $logyear-01-01 --before $logyear-12-31"

else 
    cmd="git lg --after $logyear-$month-01 --before $logyear-$month-31"
fi 

if [ "$rev" == "1" ]; then 
   cmd="$cmd | tail -r"
fi 


echo $cmd
eval $cmd


exit 0 
