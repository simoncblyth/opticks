#!/bin/bash -l 
grab_usage(){ cat << EOU
grab.sh
=============

Runs rsync grabbing into local directories files from a remote geocache/CSG_GGeo/ directory 
into which cxs jpg renders, json sidecars and intersect "photon" arrays are persisted.
The remote directory to grab from is configurable with envvar OPTICKS_KEYDIR_GRABBED,  eg::

   .opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1

NB to update the CSGFoundry geometry on laptop for correct labelling of volumes use::

   ./cf_grab.sh 

EOU
}

arg=${1:-all}
shift

executable=${EXECUTABLE:-CSGOptiXSimulateTest}

opticks_key_remote_dir=$(opticks-key-remote-dir)
xdir=$opticks_key_remote_dir/CSG_GGeo/$executable/   ## trailing slash to avoid duplicating path element 

from=P:$xdir
to=$HOME/$xdir

LOGDIR=/tmp/$USER/opticks/CSGOptiX/$EXECUTABLE


printf "arg                     %s \n" "$arg"
printf "EXECUTABLE              %s \n " "$EXECUTABLE"
printf "LOGDIR                  %s \n " "$LOGDIR"
printf "OPTICKS_KEY_REMOTE      %s \n " "$OPTICKS_KEY_REMOTE" 
printf "opticks_key_remote_dir  %s \n " "$opticks_key_remote_dir" 
printf "\n"
printf "xdir                    %s \n" "$xdir"
printf "from                    %s \n" "$from" 
printf "to                      %s \n" "$to" 

mkdir -p $to


find_last(){
    local msg="=== $FUNCNAME :"
    local typ=${1:-jpg}
    local last=$(ls -1rt `find ${to%/} -name "*.$typ" ` | tail -1 )
    echo $last 
}

relative_path(){
   local path=$1
   local pfx=${HOME}/${OPTICKS_KEYDIR_GRABBED}/CSG_GGeo/CSGOptiXRenderTest/
   local rel=""
   case $path in 
      ${pfx}*)  rel=${path/$pfx} ;;    
   esac
   echo $rel   
   #cvd1/70000/RichTbR1MagShBox/cam_0_t0/cxr_view___eye_1,-2.2,0__zoom_1__tmin_0.4_RichTbR1MagShBox.jpg
   #cvd1/70000/RichTbR1MagShBox/cam_0_t0/cxr_view___eye_1,-2.2,0__zoom_1__tmin_0.4_RichTbR1MagShBox.jpg
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


if [ "$arg" == "tab" ]; then

    globptn="${to}cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview*.jpg"
    refjpgpfx="/env/presentation/cxr/cxr_overview"

    ${IPYTHON:-ipython} -i $(which snap.py) --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $*

elif [ "$arg" == "tab_water" ]; then

    globptn="${to}cvd1/70000/sWaterTube/cxr_view/*/cxr_view_sWaterTube.jpg"
    ${IPYTHON:-ipython} -i $(which snap.py) --  --globptn "$globptn"  $*

elif [ "$arg" == "png" ]; then

    rsync -zarv --progress --include="*/" --include="*.png" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.png'`
    open_last png

elif [ "$arg" == "jpg" ]; then

    rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.json'`
    open_last jpg

elif [ "$arg" == "jpg_last" ]; then

    open_last jpg 

elif [ "$arg" == "mp4" ]; then

    rsync -zarv --progress --include="*/" --include="*.mp4" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.mp4' -o -name '*.json'`
    open_last mp4


elif [ "$arg" == "all" ]; then
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"

    ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `


    if [ "$EXECUTABLE" == "CSGOptiXSimulateTest" -o "$EXECUTABLE" == "CSGOptiXRenderTest" ]; then  

        last_npy=$(ls -1rt `find ${to%/} -name '*.npy' ` | tail -1 )
        last_outdir=$(dirname $last_npy)
        last_outbase=$(dirname $last_outdir)
        last_outleaf=$(basename $last_outdir)

        ## write source-able script ${EXECUTABLE}_OUTPUT_DIR.sh defining correponding envvar
        ## depending on the path of the last .npy grabbed  
        ## This is used from cxs.sh to transparently communicate the last OUTPUT_DIR 
        ## between nodes. 

        script=$LOGDIR/${EXECUTABLE}_OUTPUT_DIR.sh
        mkdir -p $(dirname $script)

        echo last_npy $last_npy 
        echo last_outdir $last_outdir 
        echo last_outbase $last_outbase
        echo last_outleaf $last_outleaf

        echo script $script
        printf "export ${EXECUTABLE}_OUTPUT_DIR=$last_outdir\n" > $script
        cat $script
   fi 

fi 

