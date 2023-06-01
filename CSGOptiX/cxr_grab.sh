#!/bin/bash -l 

usage(){ cat << EOU
cxr_grab.sh 
=============

::

   ./cxr_grab.sh grab 
   ./cxr_grab.sh open
   ./cxr_grab.sh clean 


Formerly used the below, but that hardcodes an old directory layout:: 

   EXECUTABLE=CSGOptiXRenderTest ./grab.sh $* 

EOU
}


defarg="grab_open"
arg=${1:-$defarg}
geom=V0J008
GEOM=${GEOM:-$geom}

base=/tmp/$USER/opticks/GEOM/$GEOM/CSGOptiXRenderTest
echo rsync GEOM $GEOM base $base

if [ "${arg/grab}" != "$arg" ]; then 
    echo $BASH_SOURCE grabbing from remote 
    source $OPTICKS_HOME/bin/rsync.sh $base
fi 

if [ "${arg/open}" != "$arg" ]; then 
    echo $BASH_SOURCE open : list jpg and json from base $base in reverse time order

    jpgs=($(ls -1t $(find $base -name '*.jpg')))
    jsons=($(ls -1t $(find $base -name '*.json')))

    for jpg in ${jpgs[*]}   ; do echo $jpg  ; done  
    for json in ${jsons[*]} ; do echo $json ; done  

    open ${jpgs[0]}
    # pretty print the json 
    python -c "import json ; js=json.load(open(\"${jsons[0]}\")) ; print(json.dumps(js, indent=4))" 

fi 

if [ "${arg/clean}" != "$arg" ]; then 
    echo $BASH_SOURCE clean : delete jpg and json found in base $base
    files=$(find $base -name '*.jpg' -o -name '*.json')
    for file in ${files[*]} ; do 
       echo file $file 
    done 
fi 
