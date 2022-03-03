#!/bin/bash -l

usage(){ cat << EOU

/tmp/blyth/opticks/snap/frame00000.jpg

TODO: more informative paths 


EOU
}

arg=${1:-jpg}
shift

xdir=/tmp/blyth/opticks/snap/
from=P:$xdir
to=$xdir

mkdir -p ${to}


printf "arg                    %s \n" "$arg"
printf "\n"
printf "xdir                   %s \n" "$xdir"
printf "from                   %s \n" "$from"
printf "to                     %s \n" "$to"


find_last(){
    local msg="=== $FUNCNAME :"
    local typ=${1:-jpg}
    local last=$(ls -1rt `find ${to%/} -name "*.$typ" ` | tail -1 )
    echo $last 
}

open_last(){
    local msg="=== $FUNCNAME :"
    local typ=${1:-jpg}
    local last=$(find_last $typ)
    echo $msg typ $typ last $last

    if [ "$(uname)" == "Darwin" ]; then
        open $last 
    fi  

    if [ -n "$PUB" ]; then
        if [ "$typ" == "jpg" -o "$typ" == "png" ]; then
            pub_path $last $typ
        fi  
    fi  
}

relative_path(){
   local path=$1
   local pfx=$to
   local rel=""
   case $path in
      ${pfx}*)  rel=${path/$pfx} ;;
   esac
   echo $rel   
}

pub_path(){
    local msg="=== $FUNCNAME :"
    local path=$1
    local typ=${2:-jpg}
    local rel=$(relative_path $path)
    rel=${rel/\.$typ}

    local ext=""
    if [ "$PUB" == "1" ]; then
        ext=""
    else
        ext="_${PUB}"
    fi

    local s5p=/env/presentation/${rel}${ext}.$typ
    local pub=$HOME/simoncblyth.bitbucket.io$s5p

    echo $msg path $path 
    echo $msg typ $typ 
    echo $msg rel $rel
    echo $msg s5p $s5p
    echo $msg pub $pub
    echo $msg s5p $s5p 1280px_720px 

    if [ -f "$pub" ]; then
        echo $msg published path exists already : NOT COPYING : set PUB to an ext string to distinguish the name or more permanently arrange for a different path   
    else
        local pubdir=$(dirname $pub)
        if [ ! -d "$pubdir" ]; then
            echo $msg creating pubdir $pubdir 
            mkdir -p "$pubdir"
        fi
        echo $msg copying path to pub 
        cp $path $pub
        echo $msg add s5p to s5_background_image.txt
    fi
}




if [ "$arg" == "jpg" ]; then

    rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.json'`
    open_last jpg 

elif [ "$arg" == "jpg_last" ]; then

    open_last jpg 

fi


