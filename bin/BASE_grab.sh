#!/bin/bash -l 

usage(){ cat << EOU
BASE_grab.sh 
==============

Usage::

   BASE=/directory/to/grab/from source $OPTICKS_HOME/bin/BASE_grab.sh grab
   BASE=/directory/to/grab/from source $OPTICKS_HOME/bin/BASE_grab.sh open
   BASE=/directory/to/grab/from source $OPTICKS_HOME/bin/BASE_grab.sh clean

EOU
}


vars="BASE"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/grab}" != "$arg" ]; then 
    echo $BASH_SOURCE grabbing from remote 
    source $OPTICKS_HOME/bin/rsync.sh $BASE
fi 

if [ "${arg/open}" != "$arg" ]; then 
    echo $BASH_SOURCE open : list jpg/json/log from BASE $BASE in reverse time order

    jpgs=($(ls -1t $(find $BASE -name '*.jpg')))
    jsons=($(ls -1t $(find $BASE -name '*.json')))
    logs=($(ls -1t $(find $BASE -name '*.log')))

    for jpg in ${jpgs[*]}   ; do echo $jpg  ; done  
    for json in ${jsons[*]} ; do echo $json ; done  
    for log in ${logs[*]}   ; do echo $log ; done  

    jpg0="${jpgs[0]}"
    if [ -f "$jpg0" ]; then 
        open $jpg0
    else
        echo $BASH_SOURCE : ERROR no jpg0 $jpg0 in BASE $BASE 
    fi 

    json0="${jsons[0]}"
    if [ -f "$json0" ]; then 
        python -c "import json ; js=json.load(open(\"$json0\")) ; print(json.dumps(js, indent=4))" 
    else
        echo $BASH_SOURCE : ERROR no json0 $json0 in BASE $BASE 
    fi 

fi 

if [ "${arg/clean}" != "$arg" ]; then 
    echo $BASH_SOURCE clean : delete jpg/json/log found in BASE $BASE
    files=$(find $BASE -name '*.jpg' -o -name '*.json' -o -name '*.log')
    for file in ${files[*]} ; do 
       echo file $file 
    done 
fi 
