#!/bin/bash
usage(){ cat << EOU
BASE_grab.sh 
==============

Usage::

   BASE=/directory/to/grab/from source BASE_grab.sh grab
   BASE=/directory/to/grab/from source BASE_grab.sh open
   BASE=/directory/to/grab/from source BASE_grab.sh clean

EOU
}

cd $(dirname $(realpath $BASH_SOURCE))

vars="BASE"
for var in $vars ; do printf "%20s : %s \n" "$var" "${!var}" ; done 


if [ "${arg/grab}" != "$arg" ]; then 
    echo $BASH_SOURCE grabbing from remote 
    source rsync.sh $BASE
fi 

if [ "${arg/clean}" != "$arg" ]; then 
    echo $BASH_SOURCE clean : delete jpg/json/log found in BASE $BASE
    files=$(find $BASE -name '*.jpg' -o -name '*.json' -o -name '*.log')
    for file in ${files[*]} ; do 
       echo file $file 
    done 
fi 

if [ "${arg/jstab}" != "$arg" ]; then 

    echo $BASH_SOURCE jstab     

    script=../ana/snap.py

    if command -v ${IPYTHON:-ipython} &> /dev/null 
    then
        PYTHONPATH=../.. ${IPYTHON:-ipython} --pdb -i $script 
    else
        echo $BASH_SOURCE - IPYTHON NOT AVAILABLE - TRY PYTHON 
        PYTHONPATH=../.. ${PYTHON:-python}  $script 
    fi  
fi 




if [ "${arg/open}" != "$arg" ]; then 
    echo $BASH_SOURCE open : list jpg/json/log from BASE $BASE in reverse time order

    jpgs=($(ls -1t $(find $BASE -name '*.jpg')))
    jsons=($(ls -1t $(find $BASE -name '*.json')))
    logs=($(ls -1t $(find $BASE -name '*.log')))

    if [ -n "$VERBOSE" ]; then 
        echo $BASH_SOURCE open : jpg 
        for jpg in ${jpgs[*]}   ; do echo $jpg  ; done  

        #echo $BASH_SOURCE open : json 
        #for json in ${jsons[*]} ; do echo $json ; done  

        #echo $BASH_SOURCE open : log
        #for log in ${logs[*]}   ; do echo $log ; done  
    fi 

    item=0
    ITEM=${ITEM:-$item}

    if [ "$ITEM" == "ALL" ]; then 
        echo $BASH_SOURCE open : ITEM $ITEM 
        open -n ${jpgs[*]}
    else
        jpgx="${jpgs[$ITEM]}"
        if [ -f "$jpgx" ]; then 
            echo $BASH_SOURCE open : jpgx $jpgx ITEM $ITEM
            open $jpgx
        else
            echo $BASH_SOURCE : ERROR no jpgx $jpgx ITEM $ITEM in BASE $BASE 
        fi 

        echo $BASH_SOURCE open : jsonx $jsonx ITEM $ITEM
        jsonx="${jsons[$ITEM]}"
        if [ -f "$jsonx" ]; then 
            python -c "import json ; js=json.load(open(\"$jsonx\")) ; print(json.dumps(js, indent=4))" 
        else
            echo $BASH_SOURCE : ERROR no jsonx $jsonx in BASE $BASE ITEM $ITEM 
        fi 
    fi
fi 


if [ "${arg/pub}" != "$arg" -o "${arg/list}" != "$arg"  ]; then

   PBAS=${BASE/GEOM*}  # portion of BASE prior to GEOM eg "/tmp/$USER/opticks/"  
   nbas=${#PBAS}    
   RBAS=${BASE:$nbas}  # BASE with PBAS removed giving relative prefix starting with GEOM
   echo $BASH_SOURCE pub BASE $BASE PBAS $PBAS nbas $nbas

   echo $BASE \#BASE
   echo $PBAS \#PBAS
   echo $RBAS \#RBAS
   echo $nbas \#nbas

   if [ $nbas -eq 0 ]; then  
       echo $BASH_SOURCE : ERROR UNEXPECTED nbas $nbas FOR BASE $BASE PBAS $PBAS 
   else
       allpaths=($(ls -1t ${BASE}/*.jpg))

       if [ "$ITEM" == "ALL" ]; then 
           paths=(${allpaths[@]})
       else
           paths=(${allpaths[$ITEM]})
       fi  

       idx=0 
       for path in ${paths[@]} ; do 
           nam=$(basename $path)   
           stem=${nam//.jpg*}
           ext=.jpg
           rel=${path:$nbas}
           path2=${PBAS}${RBAS}/${stem}${ext}
           if [ "$path2" == "$path" ]; then 
               match=Y
           else
               match=N
               echo pp $path
               echo p2 $path2
           fi
           printf "%3d : %s : %-100s \n" "$idx" "$match" "$path" 

           idx=$(( $idx + 1 ))


           if [ "${arg/pub}" != "$arg" ]; then 
               s5r=${RBAS}/${stem}_${PUB:-MISSING_PUB}${ext}
               s5p=/env/presentation/${s5r}
               s5d=$(dirname $s5p)

               cmds=("mkdir -p $s5d"  "cp $path $s5p" "echo s5p $s5r 1280px_720px")
               for cmd in "${cmds[@]}" ; do echo "$cmd" ; done 

               if [ ${#PUB} -gt 1  ]; then
                   for cmd in "${cmds[@]}" 
                   do 
                       echo "$cmd"
                       eval "$cmd"
                       [ $? -ne 0 ] && echo $BASH_SOURCE ERROR FROM cmd "$cmd" && exit 1  
                   done 
               else
                   echo rerun with PUB defined to eval the above commands 
               fi
           fi
       done  
       echo select with ITEM=0 ITEM=1 ... OR ITEM=ALL
   fi 
fi

