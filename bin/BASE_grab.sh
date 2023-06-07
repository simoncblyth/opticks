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


if [ "${arg/jstab}" != "$arg" ]; then 

    echo $BASH_SOURCE jstab     
    #jsons=($(ls -1t $(find $BASE -name '*.json')))
    #for json in ${jsons[*]} ; do echo $json ; done  

    globptn="$BASE/cxr_overview*elv*.jpg"
    refjpgpfx="/env/presentation/cxr/cxr_overview"

    ${IPYTHON:-ipython} --pdb -i $OPTICKS_HOME/ana/snap.py --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $SNAP_ARGS
fi 

if [ "${arg/pub}" != "$arg" -o "${arg/list}" != "$arg"  ]; then

   echo $BASH_SOURCE pub PPFX $PPFX PBAS $PBAS

   nbas=${#PBAS}       ## PBAS is the head of the paths to be removed eg /tmp/$USER/opticks/
   npfx=${#PPFX}       ## PPFX is path prefix of the files of interest starting with PBAS
   rpfx=${PPFX:$nbas}  ## PPFX with PBAS removed giving relative prefix 

   echo $PBAS \# PBAS
   echo $PPFX \# PPFX
   echo $rpfx \# rpfx
   echo $npfx \# npfx
   echo $nbas \# nbas

   if [ $nbas -eq 0 -o $npfx -eq 0 ]; then  
       echo $BASH_SOURCE : missing PBAS $PBAS or PPFX $PPFX
   else
       paths=($(ls -1t ${PPFX}*))

       for path in ${paths[*]} ; do 
           ext=${path:$npfx}    ## ext is the path beyond ppfx eg: .jpg .json .log .npy _meta.txt 
           rel=${path:$nbas}
           path2=${PBAS}${rpfx}${ext}
           if [ "$path2" == "$path" ]; then 
               match=Y
           else
               match=N
           fi
           printf "%-100s : %10s : %s \n" "$rel" "$ext" "$match"  

           if [ "${arg/pub}" != "$arg" -a "$ext" == ".jpg" ]; then 
               s5r=${rpfx}_${PUB:-MISSING_PUB}${ext}
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
   fi 
fi

