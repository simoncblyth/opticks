#!/bin/bash -l 

grab_usage(){ cat << EOU
grab.sh
=============

Runs rsync between a remote geocache/CSG_GGeo/ directory into which cxs 
intersect "photon" arrays are persisted and local directories. 
The remote directory to grab is configurable with envvar OPTICKS_KEYDIR_GRABBED,  eg::

   .opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1

EOU
}

arg=${1:-all}
shift

executable=${EXECUTABLE:-CSGOptiXSimulateTest}
default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/3dbec4dc3bdef47884fe48af781a179d/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}

xdir=$opticks_keydir_grabbed/CSG_GGeo/$executable/   ## trailing slash to avoid duplicating path element 

from=P:$xdir
to=$HOME/$xdir

LOGDIR=/tmp/$USER/opticks/CSGOptiX/$EXECUTABLE


printf "arg                    %s \n" "$arg"
printf "EXECUTABLE             %s \n " "$EXECUTABLE"
printf "LOGDIR                 %s \n " "$LOGDIR"
printf "OPTICKS_KEYDIR_GRABBED %s \n " "$OPTICKS_KEYDIR_GRABBED" 
printf "opticks_keydir_grabbed %s \n " "$opticks_keydir_grabbed" 
printf "\n"
printf "xdir                   %s \n" "$xdir"
printf "from                   %s \n" "$from" 
printf "to                     %s \n" "$to" 

mkdir -p $to

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

elif [ "$arg" == "jpg" ]; then
    rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.json'`

elif [ "$arg" == "mp4" ]; then
    rsync -zarv --progress --include="*/" --include="*.mp4" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.mp4' -o -name '*.json'`

elif [ "$arg" == "all" ]; then
    rsync -zarv --progress --include="*/" --include="*.txt" --include="*.npy" --include="*.jpg" --include="*.mp4" --include "*.json" --exclude="*" "$from" "$to"

    ls -1rt `find ${to%/} -name '*.json' -o -name '*.txt' `
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.mp4' -o -name '*.npy'  `


    ## write source-able script ${EXECUTABLE}_OUTPUT_DIR.sh defining correponding envvar
    ## depending on the path of the last .npy grabbed  
    ## This is used from cxs.sh to transparently communicate the last OUTPUT_DIR 
    ## between nodes. 
     
    last_npy=$(ls -1rt `find ${to%/} -name '*.npy' ` | tail -1 )
    last_outdir=$(dirname $last_npy)
    script_outdir=$LOGDIR/${EXECUTABLE}_OUTPUT_DIR.sh
    echo last_npy $last_npy 
    echo last_outdir $last_outdir 
    echo script_outdir $script_outdir
    printf "export ${EXECUTABLE}_OUTPUT_DIR=$last_outdir\n" > $script_outdir
    cat $script_outdir

fi 


