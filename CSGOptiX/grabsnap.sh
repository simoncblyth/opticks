#!/bin/bash -l 
usage(){ cat << EOU
grabsnap.sh
=============

Making table from .json sidecars grabbed from remote::

   ./grabsnap.sh           ## txt table 
   ./grabsnap.sh  --rst    ## RST table

EOU
}






executable=${EXECUTABLE:-CSGOptiXRenderTest}
default_opticks_keydir_grabbed=.opticks/geocache/DetSim0Svc_pWorld_g4live/g4ok_gltf/1ad3e6c8947a2b32dea175bc67816952/1
opticks_keydir_grabbed=${OPTICKS_KEYDIR_GRABBED:-$default_opticks_keydir_grabbed}
## OPTICKS_KEYDIR_GRABBED is set in ~/.opticksdev_config

xdir=$opticks_keydir_grabbed/CSG_GGeo/$executable/   ## trailing slash to avoid duplicating path element 
from=P:$xdir
to=$HOME/$xdir
mkdir -p $to

printf "arg                    %s \n" "$arg"
printf "EXECUTABLE             %s \n " "$EXECUTABLE"
printf "LOGDIR                 %s \n " "$LOGDIR"
printf "OPTICKS_KEYDIR_GRABBED %s \n " "$OPTICKS_KEYDIR_GRABBED" 
printf "opticks_keydir_grabbed %s \n " "$opticks_keydir_grabbed" 
printf "\n"
printf "xdir                   %s \n" "$xdir"
printf "from                   %s \n" "$from" 
printf "to                     %s \n" "$to" 

if [ "$arg" == "grab" ]; then 

    rsync -zarv --progress --include="*/" --include="*.jpg" --include="*.json" --exclude="*" "$from" "$to"
    ls -1rt `find ${to%/} -name '*.jpg' -o -name '*.json'`
fi 


globptn="${to}cvd1/70000/cxr_overview/cam_0_tmin_0.4/cxr_overview*.jpg"
refjpgpfx="/env/presentation/cxr/cxr_overview"

${IPYTHON:-ipython} -i $(which snap.py) --  --globptn "$globptn" --refjpgpfx "$refjpgpfx" $*


